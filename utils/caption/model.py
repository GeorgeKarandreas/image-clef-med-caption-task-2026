"""Model Architecture file for Caption task."""

import timm
import torch
import torch.nn as nn


from open_clip import create_model_from_pretrained
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


DEFAULT_T5_LORA_TARGET_MODULES = ("q", "v")
EXPANDED_T5_LORA_TARGET_MODULES = ("q", "k", "v", "o")
EXPANDED_T5_LORA_TARGET_MODULES_WITH_FFN = (
    *EXPANDED_T5_LORA_TARGET_MODULES,
    "wi_0",
    "wi_1",
    "wo",
)


class SimpleQFormer(nn.Module):
    """
    Learns query tokens that attend to visual tokens.

    Input:
        image_tokens: [B, N, input_dim]

    Output:
        visual_memory: [B, num_query_tokens, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_query_tokens: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_dim) * 0.02
        )

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )

        self.qformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )


    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = image_tokens.size(0)

        memory = self.input_projection(image_tokens)

        queries = self.query_tokens.expand(batch_size, -1, -1)

        visual_memory = self.qformer(
            tgt=queries,
            memory=memory,
        )

        return visual_memory


class BiomedCLIPSwinQFormerT5(nn.Module):
    def __init__(
        self,
        t5_name: str = "google/flan-t5-base",
        swin_name: str = "swin_small_patch4_window7_224",
        qformer_tokens: int = 32,
        freeze_biomedclip: bool = True,
        freeze_swin: bool = True,
        use_lora: bool = True,
        lora_target_modules: tuple[str, ...] | None = None,
    ):
        super().__init__()

        self.freeze_biomedclip = freeze_biomedclip
        self.freeze_swin = freeze_swin

        self.lora_target_modules = tuple(
            DEFAULT_T5_LORA_TARGET_MODULES
            if lora_target_modules is None else
            dict.fromkeys(lora_target_modules)
        )

        # Text decoder: T5
        self.tokenizer = AutoTokenizer.from_pretrained(t5_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)

        self.t5_hidden_size = self.t5.config.d_model
        self.t5.config.use_cache = False

        # BiomedCLIP encoder
        self.biomedclip, self.biomedclip_preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

        # BiomedCLIP encode_image usually gives a pooled 512-dim embedding.
        self.biomed_dim = 512

        # Swin-small encoder
        self.swin = timm.create_model(
            swin_name,
            pretrained=True,
            features_only=True,
            out_indices=(-1,),
        )

        swin_channels = self.swin.feature_info.channels()[-1]

        # Project both encoders to T5 hidden size
        self.biomed_proj = nn.Linear(self.biomed_dim, self.t5_hidden_size)
        self.swin_proj = nn.Linear(swin_channels, self.t5_hidden_size)
        self.biomed_norm = nn.LayerNorm(self.t5_hidden_size)
        self.swin_norm = nn.LayerNorm(self.t5_hidden_size)

        # Q-Former bridge
        self.qformer = SimpleQFormer(
            input_dim=self.t5_hidden_size,
            hidden_dim=self.t5_hidden_size,
            num_query_tokens=qformer_tokens,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
        )

        # Freeze encoders first
        if self.freeze_biomedclip:
            for p in self.biomedclip.parameters():
                p.requires_grad = False
            self.biomedclip.eval()

        if self.freeze_swin:
            for p in self.swin.parameters():
                p.requires_grad = False
            self.swin.eval()

        # LoRA adapter to T5
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=list(self.lora_target_modules),
                bias="none",
            )

            self.t5 = get_peft_model(self.t5, lora_config)

    def train(self, mode: bool = True):
        super().train(mode)

        # Keep frozen encoders in eval mode even when the full model is in train mode.
        if getattr(self, "freeze_biomedclip", False):
            self.biomedclip.eval()

        if getattr(self, "freeze_swin", False):
            self.swin.eval()

        return self

    def encode_visual_tokens(
        self,
        biomed_pixels: torch.Tensor,
        swin_pixels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            visual_memory: [B, qformer_tokens, t5_hidden_size]
        """

        # BiomedCLIP global token
        if self.freeze_biomedclip:
            with torch.no_grad():
                biomed_features = self.biomedclip.encode_image(biomed_pixels)
        else:
            biomed_features = self.biomedclip.encode_image(biomed_pixels)

        biomed_features = biomed_features.float()

        biomed_token = self.biomed_norm(
            self.biomed_proj(biomed_features)
        ).unsqueeze(1)
        # [B, 1, H]

        # Swin patch tokens
        if self.freeze_swin:
            with torch.no_grad():
                swin_feature_map = self.swin(swin_pixels)[-1]
        else:
            swin_feature_map = self.swin(swin_pixels)[-1]

        if swin_feature_map.ndim != 4:
            raise ValueError(f"Expected 4D Swin feature map, got {swin_feature_map.shape}")

        expected_channels = self.swin.feature_info.channels()[-1]

        # timm may return [B, C, H, W] or [B, H, W, C], depending on model.
        if swin_feature_map.shape[1] == expected_channels:
            swin_feature_map = swin_feature_map.permute(0, 2, 3, 1)
            # [B, C, H, W] -> [B, H, W, C]

        batch_size, height, width, channels = swin_feature_map.shape

        swin_tokens = swin_feature_map.reshape(
            batch_size,
            height * width,
            channels,
        )
        # [B, N, C]

        swin_tokens = self.swin_norm(self.swin_proj(swin_tokens))
        # [B, N, H]

        # Fuse encoders
        image_tokens = torch.cat(
            [biomed_token, swin_tokens],
            dim=1,
        )
        # [B, 1 + N, H]

        # Q-Former bridge
        visual_memory = self.qformer(image_tokens)
        # [B, 32, H]

        return visual_memory

    def forward(
        self,
        biomed_pixels: torch.Tensor,
        swin_pixels: torch.Tensor,
        labels: torch.Tensor,
    ):
        visual_memory = self.encode_visual_tokens(
            biomed_pixels=biomed_pixels,
            swin_pixels=swin_pixels,
        )

        encoder_outputs = BaseModelOutput(
            last_hidden_state=visual_memory,
        )

        encoder_attention_mask = torch.ones(
            visual_memory.shape[:2],
            dtype=torch.long,
            device=visual_memory.device,
        )

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
        )

        return outputs

    @torch.no_grad()
    def generate_caption(
        self,
        biomed_pixels: torch.Tensor,
        swin_pixels: torch.Tensor,
        max_new_tokens: int = 64,
        num_beams: int = 3,
    ):
        self.eval()

        visual_memory = self.encode_visual_tokens(
            biomed_pixels=biomed_pixels,
            swin_pixels=swin_pixels,
        )

        encoder_outputs = BaseModelOutput(
            last_hidden_state=visual_memory,
        )

        encoder_attention_mask = torch.ones(
            visual_memory.shape[:2],
            dtype=torch.long,
            device=visual_memory.device,
        )

        generated_ids = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=4,
            length_penalty=1.0,
        )

        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

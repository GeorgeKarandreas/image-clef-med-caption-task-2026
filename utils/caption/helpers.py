"""Helper utilities for caption task"""

from pathlib import Path
import re
import unicodedata

import torch


def clean_caption(text):
    """
    Cleans caption text

    Args:
        text (str): A caption.

    Returns:
        text (str): A standardized cleaned caption.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)        # Unicode normalization

    text = text.replace("\n", " ").replace("\t", " ") # Remove newlines and tabs
    text = re.sub(r"http\S+|www\.\S+", "", text)      # Remove URLs
    text = re.sub(r"\s+", " ", text)                  # Normalize whitespace
    text = re.sub(r"\s+([.,!?])", r"\1", text)        # Normalize whitespace around punctiation
    text = text.strip()                               # strip trailing spaces

    return text


def debug_model_modes(model):
    model.train()

    print("model.training:", model.training)
    print("biomedclip.training:", model.biomedclip.training)
    print("swin.training:", model.swin.training)
    print("qformer.training:", model.qformer.training)
    print("t5.training:", model.t5.training)

    assert model.training is True

    if model.freeze_biomedclip:
        assert model.biomedclip.training is False

    if model.freeze_swin:
        assert model.swin.training is False

    # LoRA, T5 trainability check
    t5_trainable = [
        name
        for name, param in model.t5.named_parameters()
        if param.requires_grad
    ]

    lora_trainable = [
        name
        for name in t5_trainable
        if "lora" in name.lower()
    ]

    non_lora_trainable = [
        name
        for name in t5_trainable
        if "lora" not in name.lower()
    ]

    print(f"t5 trainable params count: {len(t5_trainable)}")
    print(f"t5 LoRA trainable params count: {len(lora_trainable)}")
    print(f"t5 non-LoRA trainable params count: {len(non_lora_trainable)}")

    if hasattr(model.t5, "print_trainable_parameters"):
        model.t5.print_trainable_parameters()

    assert len(lora_trainable) > 0, "No trainable LoRA parameters found in T5."

    assert len(non_lora_trainable) == 0, (
        "Unexpected non-LoRA trainable T5 parameters:\n"
        + "\n".join(non_lora_trainable[:50])
    )


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {100 * trainable / total:.2f}%")


@torch.no_grad()
def debug_shapes(model, loader, device):
    model.eval()

    batch = next(iter(loader))

    biomed_pixels = batch["biomed_pixels"].to(device)
    swin_pixels = batch["swin_pixels"].to(device)
    labels = batch["labels"].to(device)

    print("biomed_pixels:", biomed_pixels.shape)
    print("swin_pixels:", swin_pixels.shape)
    print("labels:", labels.shape)

    visual_memory = model.encode_visual_tokens(
        biomed_pixels=biomed_pixels,
        swin_pixels=swin_pixels,
    )

    print("visual_memory:", visual_memory.shape)

    outputs = model(
        biomed_pixels=biomed_pixels,
        swin_pixels=swin_pixels,
        labels=labels,
    )

    print("loss:", outputs.loss.item())
    print("logits:", outputs.logits.shape)


def preprocess_to_tensor(preprocess, image):
    """
    Handles both:
    1. torchvision/open_clip/timm transforms -> Tensor [C, H, W]
    2. Hugging Face image processors -> dict/BatchFeature with pixel_values [1, C, H, W]
    """
    try:
        output = preprocess(image)

        # Case 1: OpenCLIP/timm/torchvision transform
        if isinstance(output, torch.Tensor):
            return output

        # Case 2: HF processor called without return_tensors may still return dict-like
        if isinstance(output, dict) or hasattr(output, "data"):
            pixel_values = output["pixel_values"]

            if isinstance(pixel_values, torch.Tensor):
                return pixel_values.squeeze(0)

            return torch.tensor(pixel_values).squeeze(0)

    except TypeError:
        # Case 3: HF AutoImageProcessor usually wants return_tensors="pt"
        output = preprocess(
            images=image,
            return_tensors="pt",
        )

        return output["pixel_values"].squeeze(0)

    raise TypeError(
        f"Unsupported preprocess output type: {type(output)}"
    )


def resolve_checkpoint_path(checkpoint_path: Path | None, model_dir: Path) -> Path:
    """Resolve the checkpoint path or pick the newest saved caption checkpoint."""
    if checkpoint_path is not None:
        resolved = checkpoint_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found at {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Checkpoint path must be a file: {resolved}")
        return resolved

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Pass --checkpoint explicitly."
        )

    candidates = sorted(
        [path for path in model_dir.iterdir() if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files found in {model_dir}. Pass --checkpoint explicitly."
        )

    return candidates[0].resolve()


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    """Load a checkpoint file and extract a model state_dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Expected checkpoint to contain a state_dict, got {type(state_dict)}"
        )

    if any(name.startswith("module.") for name in state_dict):
        state_dict = {
            name.removeprefix("module."): value
            for name, value in state_dict.items()
        }

    return state_dict


def normalize_caption_for_submission(caption: str) -> str:
    """Normalize generated text into a single-line CSV-safe caption string."""
    if not isinstance(caption, str):
        return ""

    return " ".join(caption.replace("\r", " ").replace("\n", " ").split())


def write_caption_submission_csv(image_ids, captions, output_path: Path):
    """Write the official caption submission CSV in evaluator-compatible format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        csvfile.write("ID,Caption\n")

        for image_id, caption in zip(image_ids, captions):
            normalized_caption = normalize_caption_for_submission(caption)
            escaped_caption = normalized_caption.replace('"', '""')
            csvfile.write(f'{image_id},"{escaped_caption}"\n')

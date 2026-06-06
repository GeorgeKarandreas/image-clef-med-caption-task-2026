"""Synthetical Caption Task full train"""

# Standard library
import math
import sys
from datetime import datetime
from pathlib import Path
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from open_clip import create_model_from_pretrained
from transformers import AutoTokenizer
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.shared.helpers import (
    is_csv_valid,
    set_seed,
    append_run_tracking_csv
)

from utils.shared.datasets import CaptionsDataset
from utils.shared.early_stopper import EarlyStopper
from utils.caption.evaluation import evaluate_captions, evaluate_loss
from utils.caption.helpers import (
    debug_model_modes,
    debug_shapes,
    print_trainable_parameters,
)
from utils.caption.image_transforms import SquarePad
from utils.caption.model import (
    BiomedCLIPSwinQFormerT5,
    EXPANDED_T5_LORA_TARGET_MODULES,
)
from utils.caption.optimization import create_caption_optimizer_and_scheduler
from utils.caption.train import train_one_epoch

IMG_DIR_PATH = PROJECT_ROOT / "data" / "dev_caption_synth" / "images"
ANNOTATIONS_FILE_PATH = PROJECT_ROOT / "data" / "dev_caption_synth" / "captions.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "caption_synth"
RESULTS_DIR = PROJECT_ROOT / "results" / "caption_synth"
RUN_TRACKING_PATH = RESULTS_DIR / "training_runs.csv"

T5_NAME = "google/flan-t5-base"
SWIN_NAME = "swin_small_patch4_window7_224"

MAX_LENGTH = 80
BATCH_SIZE = 24
NUM_WORKERS = 6
NUM_EPOCHS = 26
EARLY_STOPPING_PATIENCE = 3
SEED = 42
USE_AMP = True
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_RATIO = 0.05
MIN_WARMUP_EPOCHS = 0
LORA_TARGET_MODULES = EXPANDED_T5_LORA_TARGET_MODULES
UNFREEZE_LAST_SWIN_STAGES_AFTER_WARMUP = 0

def debug_prefix() -> str:
    """Return the debug prefix with the current time."""
    return f"[debug ({datetime.now().strftime('%H:%M:%S')})]"

def main():
    """Main Logic"""
    run_started_at = datetime.now()
    model_name="BiomedCLIPSwinQFormerT5"
    run_id = (
        f"{run_started_at:%Y%m%d_%H%M%S}_"
        f"{model_name}_seed{SEED}"
    )
    model_path = MODEL_DIR / run_id

    set_seed(SEED)
    print(f"{debug_prefix()} Seed set to {SEED}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"{debug_prefix()} Using device: {device}")

    if not is_csv_valid(ANNOTATIONS_FILE_PATH):
        raise ValueError("CSV annotation file failed validation checks")

    tokenizer = AutoTokenizer.from_pretrained(T5_NAME)

    biomedclip_model, biomed_preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    swin_for_preprocess = timm.create_model(
        SWIN_NAME,
        pretrained=True,
        features_only=True,
        out_indices=(-1,),
    )

    swin_config = resolve_model_data_config(swin_for_preprocess)

    swin_train_preprocess = create_transform(
        **swin_config,
        is_training=True,
    )

    swin_val_preprocess = create_transform(
        **swin_config,
        is_training=False,
    )

    pad_fill = 0
    affine_interpolation = InterpolationMode.BILINEAR

    shared_train_transform = transforms.Compose([
        SquarePad(fill=pad_fill),

        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=4,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                interpolation=affine_interpolation,
                fill=pad_fill,
            )
        ], p=0.5),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
            )
        ], p=0.3),

        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0),
            )
        ], p=0.1),
    ])
    shared_val_transform = transforms.Compose([
        SquarePad(fill=pad_fill),
    ])

    train_ds = CaptionsDataset(
        ANNOTATIONS_FILE_PATH,
        IMG_DIR_PATH,
        split="train",
        transform=shared_train_transform,
        biomed_preprocess=biomed_preprocess,
        swin_preprocess=swin_train_preprocess,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    print(f"{debug_prefix()} Train samples: {len(train_ds)}")

    val_ds = CaptionsDataset(
        ANNOTATIONS_FILE_PATH,
        IMG_DIR_PATH,
        split="valid",
        transform=shared_val_transform,
        biomed_preprocess=biomed_preprocess,
        swin_preprocess=swin_val_preprocess,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    print(f"{debug_prefix()} Val samples: {len(val_ds)}")

    sample = train_ds[0]

    print(sample["biomed_pixels"].shape)
    print(sample["swin_pixels"].shape)
    print(sample["labels"].shape)
    print(sample["caption"])
    print(sample["image_id"])

    use_amp = USE_AMP and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(
        f"{debug_prefix()} AMP enabled: {use_amp} | "
        f"AMP dtype: {amp_dtype} | "
        f"GradScaler enabled: {scaler.is_enabled()}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    print(f"{debug_prefix()} Train batches per epoch: {len(train_loader)}")

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    print(f"{debug_prefix()} Val batches per epoch: {len(val_loader)}")

    model = BiomedCLIPSwinQFormerT5(
        t5_name=T5_NAME,
        swin_name=SWIN_NAME,
        qformer_tokens=32,
        freeze_biomedclip=True,
        freeze_swin=True,
        use_lora=True,
        lora_target_modules=LORA_TARGET_MODULES,
    ).to(device)

    debug_model_modes(model)
    print_trainable_parameters(model)
    debug_shapes(model, train_loader, device)

    optimizer, scheduler = create_caption_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        bridge_lr=3e-5,
        lora_lr=2e-5,
        encoder_lr=1e-6,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        min_warmup_epochs=MIN_WARMUP_EPOCHS,
        min_lr=1e-7,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        include_frozen_swin_params=UNFREEZE_LAST_SWIN_STAGES_AFTER_WARMUP > 0,
    )

    swin_unfreeze_epoch = max(
        MIN_WARMUP_EPOCHS,
        math.ceil(NUM_EPOCHS * WARMUP_RATIO),
    ) + 1

    if UNFREEZE_LAST_SWIN_STAGES_AFTER_WARMUP > 0:
        print(
            f"{debug_prefix()} Scheduled Swin unfreeze: last "
            f"{UNFREEZE_LAST_SWIN_STAGES_AFTER_WARMUP} stage(s) at epoch {swin_unfreeze_epoch}"
        )

    early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE, mode="min")

    best_val_loss = float("inf")
    best_val_metrics = {
        "bert": None,
        "rouge": None,
        "bleurt": None,
    }
    best_epoch = None
    epochs_completed = 0

    for epoch in range(NUM_EPOCHS):
        print(f"{debug_prefix()} Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        epochs_completed = epoch + 1

        if (
            UNFREEZE_LAST_SWIN_STAGES_AFTER_WARMUP > 0
            and epoch + 1 == swin_unfreeze_epoch
        ):
            unfrozen_stage_names = model.unfreeze_last_swin_stages(
                UNFREEZE_LAST_SWIN_STAGES_AFTER_WARMUP
            )
            print(
                f"{debug_prefix()} Unfroze Swin stage(s): "
                + ", ".join(unfrozen_stage_names)
            )
            print_trainable_parameters(model)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_scaler=use_scaler,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        print(
            f"{debug_prefix()} Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        """
        val_metrics = evaluate_captions(
            model=model,
            loader=val_loader,
            device=device,
        )

        print(
            f"{debug_prefix()} BERT: {val_metrics['bert']:.4f} | "
            f"ROUGE: {val_metrics['rouge']:.4f} | "
            f"BLEURT: {val_metrics['bleurt']:.4f}"
        )
        """

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #best_val_metrics = dict(val_metrics)
            best_epoch = epoch + 1

        if early_stopper.step(val_loss, model):
            print(
                f"{debug_prefix()} Early stopping at epoch {epoch + 1} | "
                f"Best kept Val Loss: {early_stopper.best_value:.4f}"
            )
            break

        gc.collect()
        torch.cuda.empty_cache()

    early_stopper.restore_best_weights(model)

    # Save Best Model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    print(f"{debug_prefix()} Best model saved to {model_path}")

    run_completed_at = datetime.now()
    append_run_tracking_csv(
        RUN_TRACKING_PATH,
        {
            "created_at": run_completed_at.isoformat(timespec="seconds"),
            "run_started_at": run_started_at.isoformat(timespec="seconds"),
            "run_completed_at": run_completed_at.isoformat(timespec="seconds"),
            "run_id": run_id,
            "model_name": model_name,
            "best_epoch": best_epoch,
            "epochs_completed": epochs_completed,
            "val_loss": f"{best_val_loss:.6f}" if best_val_loss is not None else "",
            "bert": (
                f"{best_val_metrics['bert']:.6f}"
                if best_val_metrics["bert"] is not None else ""
            ),
            "rouge": (
                f"{best_val_metrics['rouge']:.6f}"
                if best_val_metrics["rouge"] is not None else ""
            ),
            "bleurt": (
                f"{best_val_metrics['bleurt']:.6f}"
                if best_val_metrics["bleurt"] is not None else ""
            ),
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": SEED,
            "use_amp": use_amp,
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "device": device.type,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "criterion": "T5ForConditionalGeneration built-in CrossEntropyLoss(ignore_index=-100)",
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__,
            "lora_target_modules": ",".join(LORA_TARGET_MODULES),
            "delayed_swin_unfreeze_stages": UNFREEZE_LAST_SWIN_STAGES_AFTER_WARMUP,
            "delayed_swin_unfreeze_epoch": (
                swin_unfreeze_epoch if UNFREEZE_LAST_SWIN_STAGES_AFTER_WARMUP > 0 else ""
            ),
            "model_path": str(model_path),
        },
    )
    print(f"{debug_prefix()} Run tracking appended to {RUN_TRACKING_PATH}")

if __name__ == "__main__":
    try:
        main()
        print(f"{debug_prefix()} Code ran")
    except (ValueError, FileNotFoundError) as e:
        print(f"{debug_prefix()} Error: {e}")
        raise

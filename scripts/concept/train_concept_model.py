"""Concept Task full train"""

# Standard library
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.shared.helpers import (
    append_run_tracking_csv,
    is_csv_valid,
    set_seed,
)
from utils.concept.helpers import (
    build_universal_label_dict,
    compute_pos_weights,
    write_concept_val_submission_csv,
)

from utils.concept.image_transforms import load_transforms
from utils.shared.datasets import ConceptsDataset
from utils.concept.models import get_model_weights, load_model
from utils.concept.evaluation import validate_one_epoch
from utils.concept.optimization import create_optimizer_and_scheduler
from utils.concept.train import train_one_epoch
from utils.shared.early_stopper import EarlyStopper
from utils.concept.metrics import AsymmetricLoss

IMG_DIR_PATH = PROJECT_ROOT / "data" / "dev_concept" / "images"
ANNOTATIONS_FILE_PATH = PROJECT_ROOT / "data" / "dev_concept" / "concepts.csv"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "concept"
RUN_TRACKING_PATH = RESULTS_DIR / "training_runs.csv"

BATCH_SIZE = 32
NUM_WORKERS = 6
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 7
DEFAULT_SEED = 42
DEFAULT_PREDICTION_THRESHOLD = 0.6
USE_AMP = True

HEAD_LEARNING_RATE = 1e-4
BACKBONE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
WARMUP_RATIO = 0.05
MIN_WARMUP_EPOCHS = 1
MIN_LR = 1e-6


def debug_prefix() -> str:
    """Return the debug prefix with the current time."""
    return f"[debug ({datetime.now().strftime('%H:%M:%S')})]"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a concept model.")
    parser.add_argument(
        "model_name",
        help="Model name to train, e.g. convnext, convnext-tiny, resnet50, tresnet, dinov2.",
    )
    parser.add_argument(
        "image_size",
        type=int,
        help="Final square image size used by the transforms.",
    )
    parser.add_argument(
        "seed",
        type=int,
        default=DEFAULT_SEED,
        help="Run Seed",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic cuDNN settings (default: enabled).",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=DEFAULT_PREDICTION_THRESHOLD,
        help="Decision threshold used for validation scoring and validation submission export.",
    )
    return parser.parse_args()


def main():
    """Main Logic"""
    args = parse_args()
    seed = args.seed
    model_name = args.model_name
    image_size = args.image_size
    deterministic = args.deterministic
    prediction_threshold = args.prediction_threshold
    run_started_at = datetime.now()
    run_id = (
        f"{run_started_at:%Y%m%d_%H%M%S}_"
        f"{model_name}_img{image_size}_seed{seed}"
    )
    run_dir = MODEL_DIR / model_name / run_id
    model_path = run_dir / "model.pt"

    if not 0.0 <= prediction_threshold <= 1.0:
        raise ValueError("prediction_threshold must be between 0.0 and 1.0")

    set_seed(seed, deterministic=deterministic)
    print(f"{debug_prefix()} Seed set to {seed}")
    print(f"{debug_prefix()} Deterministic: {deterministic}")
    print(f"{debug_prefix()} Run ID: {run_id}")
    print(f"{debug_prefix()} Model: {model_name} | Image size: {image_size}")
    print(f"{debug_prefix()} Prediction threshold: {prediction_threshold}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"{debug_prefix()} Using device: {device}")

    if not is_csv_valid(ANNOTATIONS_FILE_PATH):
        raise ValueError("CSV annotation file failed validation checks")

    label_dict = build_universal_label_dict(ANNOTATIONS_FILE_PATH)
    num_classes = len(label_dict)
    print(f"{debug_prefix()} Number of classes: {num_classes}")

    model_weights = get_model_weights(model_name)
    train_transform, val_test_transform = load_transforms(
        weights=model_weights,
        image_size=image_size,
    )

    train_ds = ConceptsDataset(
        ANNOTATIONS_FILE_PATH,
        IMG_DIR_PATH,
        label_dict,
        split="train",
        transform=train_transform,
    )
    val_ds = ConceptsDataset(
        ANNOTATIONS_FILE_PATH,
        IMG_DIR_PATH,
        label_dict,
        split="valid",
        transform=val_test_transform,
    )

    print(f"{debug_prefix()} Train samples: {len(train_ds)}")
    print(f"{debug_prefix()} Val samples: {len(val_ds)}")

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"{debug_prefix()} AMP enabled: {use_amp} | GradScaler enabled: {scaler.is_enabled()}")

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

    model = load_model(model_name, num_classes, device)

    higher_weight_decay_models = {
        "convnext",
        "convnext-tiny",
        "tresnet",
        "dinov2",
        "biomedclip"
    }
    use_model_specific_weight_decay = model_name in higher_weight_decay_models

    if use_model_specific_weight_decay:
        weight_decay  = 0.05
    else:
        weight_decay = WEIGHT_DECAY

    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"{debug_prefix()} Model parameters: {total_params:,}")
    print(f"{debug_prefix()} Param groups enabled: True")
    print(f"{debug_prefix()} Model-specific weight decay: {use_model_specific_weight_decay}")

    """
    pos_weights = compute_pos_weights(ANNOTATIONS_FILE_PATH, label_dict)
    print(
        f"{debug_prefix()} Pos weight stats | "
        f"min: {pos_weights.min().item():.4f} | "
        f"max: {pos_weights.max().item():.4f} | "
        f"mean: {pos_weights.mean().item():.4f}"
    )
    """

    criterion = AsymmetricLoss()
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        train_loader,
        num_epochs=NUM_EPOCHS,
        backbone_lr=BACKBONE_LEARNING_RATE,
        head_lr=HEAD_LEARNING_RATE,
        weight_decay=weight_decay,
        warmup_ratio=WARMUP_RATIO,
        min_warmup_epochs=MIN_WARMUP_EPOCHS,
        min_lr=MIN_LR,
        use_param_groups=True,
    )

    early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE, mode="max")

    best_val_loss = None
    best_val_primary_f1 = float("-inf")
    best_val_secondary_f1 = None
    best_epoch = None
    epochs_completed = 0

    for epoch in range(NUM_EPOCHS):
        epochs_completed = epoch + 1
        print(f"{debug_prefix()} Starting epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_amp=use_amp,
            scheduler=scheduler,
        )

        val_loss, val_primary_f1, val_secondary_f1 = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            label_dict,
            threshold=prediction_threshold,
            use_amp=use_amp,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{debug_prefix()} Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"LR: {current_lr:.6e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Primary F1: {val_primary_f1:.4f} | "
            f"Val Secondary F1: {val_secondary_f1:.4f}"
        )

        if val_primary_f1 > best_val_primary_f1:
            best_val_loss = val_loss
            best_val_primary_f1 = val_primary_f1
            best_val_secondary_f1 = val_secondary_f1
            best_epoch = epoch + 1

        if early_stopper.step(val_primary_f1, model):
            print(
                f"{debug_prefix()} Early stopping at epoch {epoch + 1} | "
                f"Best kept Val Primary F1: {early_stopper.best_value:.4f}"
            )
            break

    if early_stopper.best_state_dict is not None:
        early_stopper.restore_best_weights(model)

    # Save Best Model
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    print(f"{debug_prefix()} Best model saved to {model_path}")

    validation_ids = val_ds.data["ID"].tolist()
    val_submission_name = f"{run_id}_val_submission.csv"

    write_concept_val_submission_csv(
        model,
        val_loader,
        validation_ids,
        label_dict,
        device,
        RESULTS_DIR,
        threshold=prediction_threshold,
        use_amp=use_amp,
        csv_name=val_submission_name,
    )

    print(f"{debug_prefix()} Valdiation submission saved to {RESULTS_DIR}")

    run_completed_at = datetime.now()
    append_run_tracking_csv(
        RUN_TRACKING_PATH,
        {
            "created_at": run_completed_at.isoformat(timespec="seconds"),
            "run_started_at": run_started_at.isoformat(timespec="seconds"),
            "run_completed_at": run_completed_at.isoformat(timespec="seconds"),
            "run_id": run_id,
            "model_name": model_name,
            "image_size": image_size,
            "best_epoch": best_epoch,
            "epochs_completed": epochs_completed,
            "val_loss": f"{best_val_loss:.6f}" if best_val_loss is not None else "",
            "primary_f1": f"{best_val_primary_f1:.6f}" if best_epoch is not None else "",
            "secondary_f1": f"{best_val_secondary_f1:.6f}" if best_val_secondary_f1 is not None else "",
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": seed,
            "deterministic": deterministic,
            "prediction_threshold": prediction_threshold,
            "use_amp": use_amp,
            "device": device.type,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "num_classes": num_classes,
            "num_parameters": total_params,
            "learning_rate": None,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "backbone_learning_rate": BACKBONE_LEARNING_RATE,
            "weight_decay": weight_decay,
            "warmup_ratio": WARMUP_RATIO,
            "min_warmup_epochs": MIN_WARMUP_EPOCHS,
            "min_lr": MIN_LR,
            "criterion": type(criterion).__name__,
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__,
            "weight_decay_param_groups": True,
            "model_specific_weight_decay": use_model_specific_weight_decay,
            "pretrained_weights": str(model_weights) if model_weights is not None else "",
            "model_path": str(model_path),
            "val_submission_csv": val_submission_name,
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

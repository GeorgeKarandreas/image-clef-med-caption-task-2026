"""Weighted soft-vote ensemble of concept task models on synthetical test images."""

# Standard library
import sys
from datetime import datetime
from pathlib import Path

import torch

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.shared.helpers import (
    is_csv_valid,
    set_seed,
)
from utils.concept.helpers import (
    build_test_loader,
    build_universal_label_dict,
    infer_checkpoint_metadata,
    write_soft_vote_submission_csv,
)
from utils.concept.models import load_model
from utils.concept.evaluation import collect_test_probabilities


IMG_DIR_PATH = PROJECT_ROOT / "data" / "test_synth" / "images"
ANNOTATIONS_FILE_PATH = PROJECT_ROOT / "data" / "dev_concept_synth" / "concepts.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "concept_synth"
MODELS_DIR = PROJECT_ROOT / "models" / "concept_synth"

BATCH_SIZE = 32
NUM_WORKERS = 6
SEED = 42
DEFAULT_PREDICTION_THRESHOLD = 0.6
USE_AMP = True
DETERMINISTIC = True

MODEL_PATHS = [
    MODELS_DIR / "convnext" / "20260504_002243_convnext_img224_seed42" / "model.pt",
    MODELS_DIR / "swinsmall" / "20260503_161128_swinsmall_img224_seed42" / "model.pt",
]

MODEL_WEIGHTS = [0.4, 0.6]


def debug_prefix() -> str:
    """Return the debug prefix with the current time."""
    return f"[debug ({datetime.now().strftime('%H:%M:%S')})]"


def main():
    if not MODEL_PATHS:
        raise ValueError("MODEL_PATHS is empty. Add at least one checkpoint path.")

    model_paths = [Path(model_path).resolve() for model_path in MODEL_PATHS]
    weights = [float(weight) for weight in MODEL_WEIGHTS]
    deterministic = DETERMINISTIC
    prediction_threshold = DEFAULT_PREDICTION_THRESHOLD

    if not 0.0 <= prediction_threshold <= 1.0:
        raise ValueError("prediction_threshold must be between 0.0 and 1.0")

    if len(weights) != len(model_paths):
        raise ValueError("MODEL_WEIGHTS must have the same length as MODEL_PATHS")

    if any(weight < 0.0 for weight in weights):
        raise ValueError("weights must be non-negative")


    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found at {model_path}")

    checkpoint_metadata = [
        infer_checkpoint_metadata(model_path) for model_path in model_paths
    ]
    unique_model_names = sorted(
        {metadata["model_name"] for metadata in checkpoint_metadata}
    )

    set_seed(SEED, deterministic=deterministic)
    print(f"{debug_prefix()} Seed set to {SEED}")
    print(f"{debug_prefix()} Deterministic: {deterministic}")
    print(f"{debug_prefix()} Ensemble size: {len(model_paths)}")
    print(f"{debug_prefix()} Models: {', '.join(unique_model_names)}")
    print(f"{debug_prefix()} Weights: {', '.join(f'{weight:.4f}' for weight in weights)}")
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
    use_amp = USE_AMP and device.type == "cuda"

    test_loader_cache = {}
    ensemble_probabilities = None
    ensemble_image_ids = None

    for index, model_path in enumerate(model_paths, start=1):
        metadata = checkpoint_metadata[index - 1]
        weight = weights[index - 1]
        model_name = metadata["model_name"]
        image_size = metadata["image_size"]
        checkpoint_seed = metadata["seed"]
        checkpoint_run_id = metadata["run_id"]
        loader_key = (model_name, image_size)

        if loader_key not in test_loader_cache:
            test_ds, test_loader = build_test_loader(
                model_name,
                image_size,
                IMG_DIR_PATH,
                BATCH_SIZE,
                NUM_WORKERS,
            )
            test_loader_cache[loader_key] = test_loader
            print(
                f"{debug_prefix()} Prepared test loader | "
                f"model: {model_name} | image size: {image_size} | "
                f"test samples: {len(test_ds)}"
            )
        else:
            test_loader = test_loader_cache[loader_key]

        print(
            f"{debug_prefix()} Loading checkpoint {index}/{len(model_paths)} | "
            f"model: {model_name} | image size: {image_size} | "
            f"seed: {checkpoint_seed} | run_id: {checkpoint_run_id} | "
            f"weight: {weight:.4f}"
        )

        model = load_model(model_name, num_classes, device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        image_ids, probabilities = collect_test_probabilities(
            model,
            test_loader,
            device,
            use_amp=use_amp,
        )

        if ensemble_probabilities is None:
            ensemble_image_ids = image_ids
            ensemble_probabilities = probabilities * weight
        else:
            if image_ids != ensemble_image_ids:
                raise ValueError(
                    "Checkpoint predictions produced inconsistent image ordering."
                )
            ensemble_probabilities += probabilities * weight

        del model

    total_weight = sum(weights)
    ensemble_probabilities /= total_weight

    csv_name = (
        f"mixed_model_vote_ensemble_test_synth_submission_{datetime.now().strftime('%H%M%S')}.csv"
        if len(model_paths) > 1
        else f"{checkpoint_metadata[0]['model_name']}_test_submission.csv"
    )
    write_soft_vote_submission_csv(
        ensemble_image_ids,
        ensemble_probabilities,
        label_dict,
        RESULTS_DIR,
        threshold=prediction_threshold,
        csv_name=csv_name,
    )

    print(f"{debug_prefix()} Test submission saved to {RESULTS_DIR}")


if __name__ == "__main__":
    try:
        main()
        print(f"{debug_prefix()} Code ran")
    except (ValueError, FileNotFoundError) as error:
        print(f"{debug_prefix()} Error: {error}")
        raise

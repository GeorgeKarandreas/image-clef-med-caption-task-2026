"""Generate a caption submission.csv for the synthetical test set."""

# Standard library
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.caption.image_transforms import CaptionTestTransform, SquarePad
from utils.caption.model import (
    BiomedCLIPSwinQFormerT5,
    EXPANDED_T5_LORA_TARGET_MODULES,
)
from utils.shared.datasets import TestImageDataset
from utils.caption.helpers import (
    load_state_dict,
    resolve_checkpoint_path,
    write_caption_submission_csv,
)
from utils.shared.helpers import set_seed


TEST_IMG_DIR = PROJECT_ROOT / "data" / "test_synth" / "images"
MODEL_DIR = PROJECT_ROOT / "models" / "caption_synth"
RESULTS_DIR = PROJECT_ROOT / "results" / "caption_synth"
OUTPUT_PATH = RESULTS_DIR / f"caption_synth_submission_{datetime.now().strftime('%H%M%S')}.csv"

T5_NAME = "google/flan-t5-base"
SWIN_NAME = "swin_small_patch4_window7_224"

BATCH_SIZE = 32
NUM_WORKERS = 6
SEED = 42
USE_AMP = True
MAX_NEW_TOKENS = 64
NUM_BEAMS = 5
LORA_TARGET_MODULES = EXPANDED_T5_LORA_TARGET_MODULES


def debug_prefix() -> str:
    """Return the debug prefix with the current time."""
    return f"[debug ({datetime.now().strftime('%H:%M:%S')})]"


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a caption submission.csv for the ImageCLEFmedical 2026 test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a saved caption checkpoint. Defaults to the newest file in models/caption.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=TEST_IMG_DIR,
        help="Directory containing the official caption test .jpg images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to the output submission.csv file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for caption generation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum number of generated tokens per caption.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=NUM_BEAMS,
        help="Beam width used during generation.",
    )
    return parser.parse_args()


def main():
    """Load a caption checkpoint, run test inference, and write submission.csv."""
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, MODEL_DIR)

    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    use_amp = USE_AMP and device.type == "cuda"

    print(f"{debug_prefix()} Using device: {device}")
    print(f"{debug_prefix()} Checkpoint: {checkpoint_path}")
    print(f"{debug_prefix()} Test image dir: {args.test_dir}")

    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {args.test_dir}")

    model = BiomedCLIPSwinQFormerT5(
        t5_name=T5_NAME,
        swin_name=SWIN_NAME,
        qformer_tokens=32,
        freeze_biomedclip=True,
        freeze_swin=True,
        use_lora=True,
        lora_target_modules=LORA_TARGET_MODULES,
    ).to(device)

    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    shared_transform = SquarePad(fill=0)
    swin_config = resolve_model_data_config(model.swin)
    swin_test_preprocess = create_transform(
        **swin_config,
        is_training=False,
    )

    test_transform = CaptionTestTransform(
        biomed_preprocess=model.biomedclip_preprocess,
        swin_preprocess=swin_test_preprocess,
        shared_transform=shared_transform,
    )

    test_ds = TestImageDataset(
        img_dir=args.test_dir,
        transform=test_transform,
    )

    if len(test_ds) == 0:
        raise ValueError(f"No .jpg files found in test image directory: {args.test_dir}")

    print(f"{debug_prefix()} Test samples: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    image_ids = []
    captions = []

    for batch_inputs, batch_image_ids in test_loader:
        biomed_pixels = batch_inputs["biomed_pixels"].to(device, non_blocking=True)
        swin_pixels = batch_inputs["swin_pixels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            generated = model.generate_caption(
                biomed_pixels=biomed_pixels,
                swin_pixels=swin_pixels,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

        image_ids.extend(batch_image_ids)
        captions.extend(generated)

    if len(image_ids) != len(captions):
        raise ValueError(
            f"Generated caption count mismatch: {len(image_ids)} ids vs {len(captions)} captions."
        )

    write_caption_submission_csv(image_ids, captions, args.output)

    print(f"{debug_prefix()} Submission saved to {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
        print(f"{debug_prefix()} Code ran")
    except (ValueError, FileNotFoundError) as e:
        print(f"{debug_prefix()} Error: {e}")
        raise

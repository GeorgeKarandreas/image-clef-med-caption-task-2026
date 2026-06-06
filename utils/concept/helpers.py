"""Python File with helper functions"""

import csv
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.concept.image_transforms import load_transforms
from utils.concept.models import get_model_weights
from utils.shared.datasets import TestImageDataset


def build_universal_label_dict(annotations_csv_path):
    """
    Builds a single universal label dictionary from the full annotations file

    Args:
        annotations_csv_path (str): Path to the CSV file.

    Returns:
        label_dictionary (dict): A standard label map for a annotations file.
    """
    df = pd.read_csv(annotations_csv_path)

    all_labels = set()

    for labels in df["CUIs"]:
        all_labels.update(labels.split(";"))

    label_list = sorted(list(all_labels))
    label_dictionary = {label: i for i, label in enumerate(label_list)}

    return label_dictionary


def compute_pos_weights(csv_file, label_map):
    """
    Compute class-wise positive weights from a CSV for imbalanced multi-label training.

    Args:
        csv_file (str): Path to the CSV file containing a "CUIs" column.
        label_map (dict): Mapping from CUI string to class index.

    Returns:
        torch.Tensor: Log-scaled positive weights for each class.
    """
    df = pd.read_csv(csv_file)

    num_classes = len(label_map)
    total_samples = len(df)

    # Count positives per class
    pos_counts = torch.zeros(num_classes)

    for cuis in df["CUIs"]:
        if pd.isna(cuis):
            continue

        for cui in cuis.split(";"):
            cui = cui.strip()
            if cui in label_map:
                pos_counts[label_map[cui]] += 1

    # Avoid division by zero
    pos_counts = torch.clamp(pos_counts, min=1)

    neg_counts = total_samples - pos_counts

    pos_weights = torch.log1p(neg_counts / pos_counts)

    pos_weights = torch.clamp(pos_weights, max=12.0)

    return pos_weights


def write_concept_test_submission_csv(
    model,
    submission_loader,
    label_dict,
    device,
    output_path,
    threshold=0.5,
    use_amp=False,
    csv_name="test_submission",
):
    """Write predictions in the official submission CSV format."""
    output_path = Path(output_path)
    if csv_name is not None:
        output_path = output_path / csv_name

    inverse_label_dict = {index: label for label, index in label_dict.items()}
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        writer.writerow(["ID", "CUIs"])

        with torch.no_grad():
            for images, image_ids in submission_loader:
                images = images.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)

                probs = torch.sigmoid(logits)
                batch_threshold = threshold

                if isinstance(batch_threshold, np.ndarray):
                    batch_threshold = torch.from_numpy(batch_threshold)

                if isinstance(batch_threshold, torch.Tensor):
                    batch_threshold = batch_threshold.to(probs.device, dtype=probs.dtype)

                predictions = (probs >= batch_threshold).int().cpu()

                for image_id, prediction_row in zip(image_ids, predictions):
                    predicted_cuis = [
                        inverse_label_dict[index]
                        for index, is_present in enumerate(prediction_row.tolist())
                        if is_present
                    ][:100]

                    writer.writerow([image_id, ";".join(predicted_cuis)])


def write_concept_val_submission_csv(
    model,
    submission_loader,
    validation_ids,
    label_dict,
    device,
    output_path,
    threshold=0.5,
    use_amp=False,
    csv_name="valid_submission",
):
    """Write a concept-task validation submission.csv from a labeled loader."""
    output_path = Path(output_path)
    if csv_name is not None:
        output_path = output_path / csv_name

    inverse_label_dict = {index: label for label, index in label_dict.items()}
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_index = 0

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        writer.writerow(["ID", "CUIs"])

        with torch.no_grad():
            for images, _ in submission_loader:
                images = images.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)

                probs = torch.sigmoid(logits)
                batch_threshold = threshold

                if isinstance(batch_threshold, np.ndarray):
                    batch_threshold = torch.from_numpy(batch_threshold)

                if isinstance(batch_threshold, torch.Tensor):
                    batch_threshold = batch_threshold.to(probs.device, dtype=probs.dtype)

                predictions = (probs >= batch_threshold).int().cpu()
                batch_image_ids = validation_ids[row_index: row_index + len(images)]

                for image_id, prediction_row in zip(batch_image_ids, predictions):
                    predicted_cuis = [
                        inverse_label_dict[index]
                        for index, is_present in enumerate(prediction_row.tolist())
                        if is_present
                    ][:100]

                    writer.writerow([image_id, ";".join(predicted_cuis)])

                row_index += len(images)


def infer_checkpoint_metadata(model_path: Path) -> dict[str, int | str]:
    """Infer checkpoint metadata from the run directory structure."""
    model_path = model_path.resolve()

    if len(model_path.parents) < 2:
        raise ValueError(
            "model_path must look like "
            "'models/<model_name>/<run_id>/model.pt'."
        )

    run_id = model_path.parent.name
    model_name = model_path.parent.parent.name
    match = re.search(r"_img(?P<image_size>\d+)_seed(?P<seed>\d+)$", run_id)

    if match is None:
        raise ValueError(
            f"Could not infer image_size and seed from run_id '{run_id}'. "
            "Expected a suffix like '_img384_seed42'."
        )

    return {
        "model_name": model_name,
        "image_size": int(match.group("image_size")),
        "seed": int(match.group("seed")),
        "run_id": run_id,
    }


def write_soft_vote_submission_csv(
    image_ids,
    probabilities,
    label_dict,
    output_path,
    threshold=0.5,
    csv_name="test_submission",
):
    """Write an official submission CSV from ensemble probabilities."""
    output_path = Path(output_path)
    if csv_name is not None:
        output_path = output_path / csv_name

    inverse_label_dict = {index: label for label, index in label_dict.items()}
    predictions = (probabilities >= threshold).int()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        writer.writerow(["ID", "CUIs"])

        for image_id, prediction_row in zip(image_ids, predictions.tolist()):
            predicted_cuis = [
                inverse_label_dict[index]
                for index, is_present in enumerate(prediction_row)
                if is_present
            ][:100]
            writer.writerow([image_id, ";".join(predicted_cuis)])


def build_test_loader(model_name, image_size, img_dir, batch_size, num_workers):
    """Build the test loader for a specific model family and image size."""
    model_weights = get_model_weights(model_name)
    _, val_test_transform = load_transforms(
        weights=model_weights,
        image_size=image_size,
    )

    test_ds = TestImageDataset(img_dir, transform=val_test_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return test_ds, test_loader

"""Validation and Inference utilities for the concept task."""

import numpy as np
import torch
from utils.concept.metrics import compute_primary_f1_score, compute_secondary_f1_score


def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    label_dict,
    threshold=0.5,
    use_amp=False,
    return_outputs=False,
):
    """Run one validation epoch and return loss plus official primary and secondary scores."""
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_batches = 0
    all_probs = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)
            batch_threshold = threshold

            if isinstance(batch_threshold, np.ndarray):
                batch_threshold = torch.from_numpy(batch_threshold)

            if isinstance(batch_threshold, torch.Tensor):
                batch_threshold = batch_threshold.to(probs.device, dtype=probs.dtype)

            predictions = (probs >= batch_threshold).int()
            target_labels = targets.int()

            total_loss += loss.detach().float()
            total_batches += 1
            all_probs.append(probs.cpu())
            all_predictions.append(predictions.cpu())
            all_targets.append(target_labels.cpu())

    if total_batches == 0:
        raise ValueError("val_loader is empty.")

    avg_loss = (total_loss / total_batches).item()
    y_prob = torch.cat(all_probs).numpy()
    y_pred = torch.cat(all_predictions).numpy()
    y_true = torch.cat(all_targets).numpy()
    primary_f1_score = compute_primary_f1_score(y_true, y_pred)
    secondary_f1_score = compute_secondary_f1_score(y_true, y_pred, label_dict)

    if return_outputs:
        return y_true, y_prob, None

    return avg_loss, primary_f1_score, secondary_f1_score


def collect_test_probabilities(model, test_loader, device, use_amp=False):
    """Run inference and return ordered image ids plus sigmoid probabilities."""
    model.eval()
    all_probs = []
    all_image_ids = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)

            all_probs.append(torch.sigmoid(logits).cpu())
            all_image_ids.extend(image_ids)

    if not all_probs:
        raise ValueError("test_loader is empty.")

    return all_image_ids, torch.cat(all_probs, dim=0)

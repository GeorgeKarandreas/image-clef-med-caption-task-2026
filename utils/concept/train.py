"""Training utilities for the concept task."""
import torch


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    use_amp=False,
    scheduler=None,
):
    """Run one training epoch and return the average training loss."""
    if len(train_loader) == 0:
        raise ValueError("train_loader is empty.")

    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_batches = 0

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.detach().float()
        total_batches += 1

    avg_loss = (total_loss / total_batches).item()

    return avg_loss

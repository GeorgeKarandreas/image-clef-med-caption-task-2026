"""Train utilities for caption task"""

from tqdm.auto import tqdm
import torch


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    scaler=None,
    use_amp=False,
    amp_dtype=torch.float16,
    use_scaler=False,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
):
    if len(loader) == 0:
        raise ValueError("loader is empty.")

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")

    if use_scaler and scaler is None:
        raise ValueError("use_scaler=True but scaler is None.")

    if use_amp and amp_dtype == torch.float16 and not use_scaler:
        raise ValueError(
            "fp16 AMP should use GradScaler. "
            "Either set use_scaler=True or use amp_dtype=torch.bfloat16."
        )

    model.train()

    total_loss = 0.0
    valid_batches = 0
    skipped_batches = 0
    optimizer_steps = 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader), start=1):
        biomed_pixels = batch["biomed_pixels"].to(device, non_blocking=True)
        swin_pixels = batch["swin_pixels"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        autocast_enabled = use_amp and device.type == "cuda"

        with torch.amp.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=amp_dtype,
        ):
            outputs = model(
                biomed_pixels=biomed_pixels,
                swin_pixels=swin_pixels,
                labels=labels,
            )

            raw_loss = outputs.loss

            # Divide loss so accumulated gradients match large-batch averaging.
            loss = raw_loss / gradient_accumulation_steps

        if not torch.isfinite(raw_loss):
            skipped_batches += 1
            print(f"Skipping batch with non-finite loss: {raw_loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += raw_loss.detach().float().item()
        valid_batches += 1

        should_step = (
            step % gradient_accumulation_steps == 0
            or step == len(loader)
        )

        if not should_step:
            continue

        if use_scaler:
            scale_before = scaler.get_scale()

            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=max_grad_norm,
            )

            if not torch.isfinite(grad_norm):
                skipped_batches += 1
                print(f"Skipping optimizer step with non-finite grad norm: {grad_norm}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            scale_after = scaler.get_scale()
            optimizer_stepped = scale_after >= scale_before

            if scheduler is not None and optimizer_stepped:
                scheduler.step()

            if optimizer_stepped:
                optimizer_steps += 1

        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=max_grad_norm,
            )

            if not torch.isfinite(grad_norm):
                skipped_batches += 1
                print(f"Skipping optimizer step with non-finite grad norm: {grad_norm}")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer_steps += 1

        optimizer.zero_grad(set_to_none=True)

    if valid_batches == 0:
        raise RuntimeError(
            f"All batches were skipped. skipped_batches={skipped_batches}"
        )

    avg_loss = total_loss / valid_batches

    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batches/steps due to NaN/Inf.")

    print(f"Optimizer steps this epoch: {optimizer_steps}")

    return avg_loss

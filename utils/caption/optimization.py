"""Optimizer and Scheduler utilities for caption task."""

import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, LambdaLR


def create_caption_optimizer_and_scheduler(
    model,
    train_loader,
    num_epochs,
    bridge_lr=1e-5,
    lora_lr=1e-5,
    encoder_lr=1e-6,
    weight_decay=0.01,
    warmup_ratio=0.10,
    min_warmup_epochs=1,
    min_lr=None,
    min_lr_ratio=0.2,
    gradient_accumulation_steps=1,
    include_frozen_swin_params=False,
):
    """
    Create AdamW optimizer with linear warmup followed by cosine decay.

    Scheduler is step-based, so  scheduler.step() once per optimizer step
    """

    if len(train_loader) == 0:
        raise ValueError("train_loader is empty.")

    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1.")

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")

    if not 0.0 < warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1.")

    if min_warmup_epochs < 0:
        raise ValueError("min_warmup_epochs must be >= 0.")

    if bridge_lr <= 0 or lora_lr <= 0 or encoder_lr <= 0:
        raise ValueError("Learning rates must be positive.")

    if min_lr is not None:
        print("Warning: min_lr is ignored. Use min_lr_ratio instead.")

    if not 0.0 <= min_lr_ratio < 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1).")

    optimizer_params = create_caption_param_groups(
        model=model,
        bridge_lr=bridge_lr,
        lora_lr=lora_lr,
        encoder_lr=encoder_lr,
        weight_decay=weight_decay,
        include_frozen_swin_params=include_frozen_swin_params,
    )

    optimizer = AdamW(
        optimizer_params,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / gradient_accumulation_steps
    )

    total_steps = num_epochs * optimizer_steps_per_epoch

    if total_steps <= 1:
        raise ValueError("total optimizer steps must be greater than 1.")

    ratio_warmup_steps = int(total_steps * warmup_ratio)
    min_warmup_steps = min_warmup_epochs * optimizer_steps_per_epoch

    warmup_steps = max(1, ratio_warmup_steps, min_warmup_steps)

    # Keep at least one cosine step.
    warmup_steps = min(warmup_steps, total_steps - 1)

    cosine_steps = total_steps - warmup_steps

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    def cosine_lambda(step: int):
        progress = step / max(1, cosine_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    cosine_scheduler = LambdaLR(
        optimizer,
        lr_lambda=cosine_lambda,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    return optimizer, scheduler


def create_caption_param_groups(
    model,
    bridge_lr: float,
    lora_lr: float,
    encoder_lr: float,
    weight_decay: float,
    include_frozen_swin_params: bool = False,
):
    """
    Create AdamW parameter groups for the caption model.

    Groups:
        bridge:
            qformer, biomed_proj, swin_proj, biomed_norm, swin_norm

        lora:
            T5 LoRA adapter params

        encoder:
            BiomedCLIP / Swin params, only relevant if unfrozen

        other:
            safety group for anything trainable but unmatched
    """

    bridge_decay = []
    bridge_no_decay = []

    lora_decay = []
    lora_no_decay = []

    encoder_decay = []
    encoder_no_decay = []

    other_decay = []
    other_no_decay = []

    no_decay_keywords = (
        "bias",
        "norm",
        "layernorm",
        "layer_norm",
        "query_tokens",
        "pos_embed",
        "position",
        "embedding",
    )

    bridge_keywords = (
        "qformer",
        "biomed_proj",
        "swin_proj",
        "biomed_norm",
        "swin_norm",
    )

    lora_keywords = (
        "lora",
    )

    encoder_keywords = (
        "biomedclip",
        "swin",
    )

    for name, param in model.named_parameters():
        name_lower = name.lower()
        is_swin_param = "swin" in name_lower

        if not param.requires_grad and not (
            include_frozen_swin_params and is_swin_param
        ):
            continue

        is_no_decay = (
            param.ndim == 1
            or name_lower.endswith(".bias")
            or any(keyword in name_lower for keyword in no_decay_keywords)
        )

        is_lora = any(keyword in name_lower for keyword in lora_keywords)
        is_bridge = any(keyword in name_lower for keyword in bridge_keywords)
        is_encoder = any(keyword in name_lower for keyword in encoder_keywords)

        # Order matters: projection and norm layers should be bridge, not encoder.
        if is_lora:
            if is_no_decay:
                lora_no_decay.append(param)
            else:
                lora_decay.append(param)

        elif is_bridge:
            if is_no_decay:
                bridge_no_decay.append(param)
            else:
                bridge_decay.append(param)

        elif is_encoder:
            if is_no_decay:
                encoder_no_decay.append(param)
            else:
                encoder_decay.append(param)

        else:
            if is_no_decay:
                other_no_decay.append(param)
            else:
                other_decay.append(param)

    param_groups = [
        {
            "name": "bridge_decay",
            "params": bridge_decay,
            "lr": bridge_lr,
            "weight_decay": weight_decay,
        },
        {
            "name": "bridge_no_decay",
            "params": bridge_no_decay,
            "lr": bridge_lr,
            "weight_decay": 0.0,
        },
        {
            "name": "lora_decay",
            "params": lora_decay,
            "lr": lora_lr,
            "weight_decay": 0.0,
        },
        {
            "name": "lora_no_decay",
            "params": lora_no_decay,
            "lr": lora_lr,
            "weight_decay": 0.0,
        },
        {
            "name": "encoder_decay",
            "params": encoder_decay,
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "name": "encoder_no_decay",
            "params": encoder_no_decay,
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "name": "other_decay",
            "params": other_decay,
            "lr": bridge_lr,
            "weight_decay": weight_decay,
        },
        {
            "name": "other_no_decay",
            "params": other_no_decay,
            "lr": bridge_lr,
            "weight_decay": 0.0,
        },
    ]

    param_groups = [group for group in param_groups if len(group["params"]) > 0]

    return param_groups

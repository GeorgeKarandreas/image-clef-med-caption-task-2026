"""Optimizer and Scheduler Code"""
import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def create_optimizer_and_scheduler(
    model,
    train_loader,
    num_epochs,
    backbone_lr=3e-5,
    head_lr=3e-4,
    weight_decay=0.05,
    warmup_ratio=0.05,
    min_warmup_epochs=2,
    min_lr=1e-6,
    use_param_groups=True,
):
    """
    Create AdamW optimizer with linear warmup followed by cosine decay.

    The scheduler is step-based, so scheduler.step() once per optimizer step.
    """

    steps_per_epoch = len(train_loader)

    if steps_per_epoch == 0:
        raise ValueError("train_loader is empty.")

    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1.")

    total_steps = num_epochs * steps_per_epoch

    if total_steps <= 1:
        raise ValueError("total_steps must be greater than 1.")

    if not 0.0 < warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1.")

    if min_warmup_epochs < 0:
        raise ValueError("min_warmup_epochs must be >= 0.")

    if backbone_lr <= 0 or head_lr <= 0:
        raise ValueError("Learning rates must be positive.")

    if min_lr >= max(backbone_lr, head_lr):
        raise ValueError("min_lr must be smaller than the largest learning rate.")

    if use_param_groups:
        optimizer_params = create_param_groups(
            model=model,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
        )

        optimizer = AdamW(optimizer_params)

    else:
        optimizer = AdamW(
            model.parameters(),
            lr=head_lr,
            weight_decay=weight_decay,
        )

    # Ratio-based warmup, but at least min_warmup_epochs.
    ratio_warmup_epochs = math.ceil(num_epochs * warmup_ratio)
    warmup_epochs = max(min_warmup_epochs, ratio_warmup_epochs)

    # Safety: warmup cannot exceed total training.
    warmup_epochs = min(warmup_epochs, num_epochs)

    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = max(1, total_steps - warmup_steps)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_lr,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    return optimizer, scheduler


def create_param_groups(
    model,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
):
    """Create AdamW parameter groups.

    Groups:
        1. backbone parameters with weight decay
        2. backbone parameters without weight decay
        3. head parameters with weight decay
        4. head parameters without weight decay

    No decay:
        - biases
        - 1D parameters, usually norm weights/biases
        - position embeddings / special tokens
        - ConvNeXt layer-scale parameters
    """

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    no_decay_names = {
        "pos_embed",
        "cls_token",
        "dist_token",
    }

    no_decay_keywords = (
        "layer_scale",
    )

    head_keywords = (
        "head",
        "classifier",
        "fc",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()

        is_head = any(keyword in name_lower for keyword in head_keywords)

        is_no_decay = (
            param.ndim == 1
            or name_lower.endswith(".bias")
            or name_lower in no_decay_names
            or any(keyword in name_lower for keyword in no_decay_keywords)
        )

        if is_head and is_no_decay:
            head_no_decay.append(param)
        elif is_head and not is_no_decay:
            head_decay.append(param)
        elif not is_head and is_no_decay:
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    param_groups = [
        {
            "params": backbone_decay,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": backbone_no_decay,
            "lr": backbone_lr,
            "weight_decay": 0.0,
        },
        {
            "params": head_decay,
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": head_no_decay,
            "lr": head_lr,
            "weight_decay": 0.0,
        },
    ]

    # Remove empty groups
    param_groups = [group for group in param_groups if len(group["params"]) > 0]

    return param_groups

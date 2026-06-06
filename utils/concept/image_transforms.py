"""Image transform utilities for the concept task."""

import torch
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as F


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 288


class ResizeLongestSide:
    """Resize an image so its longest side matches the requested size."""

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, antialias=True):
        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, image):
        height, width = F.get_size(image)
        scale = self.size / max(height, width)
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))

        return F.resize(
            image,
            size=[new_height, new_width],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )


class PadToSquare:
    """Pad an image to a square canvas using symmetric padding."""

    def __init__(self, size, fill):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        height, width = F.get_size(image)
        pad_height = max(0, self.size - height)
        pad_width = max(0, self.size - width)

        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left

        return F.pad(image, [left, top, right, bottom], fill=self.fill)



def load_transforms(weights=None, image_size=None, exact_eval_transform=False):
    """Return train and validation/test transforms."""

    if weights is not None:
        weights_transform = weights.transforms()
        mean = tuple(weights_transform.mean)
        std = tuple(weights_transform.std)
        interpolation = weights_transform.interpolation
        if image_size is None:
            image_size = weights_transform.crop_size[0]
    else:
        mean = DEFAULT_MEAN
        std = DEFAULT_STD
        interpolation = InterpolationMode.BILINEAR
        if image_size is None:
            image_size = DEFAULT_IMAGE_SIZE

    pad_fill = tuple(int(channel * 255) for channel in mean)

    affine_interpolation = interpolation

    if affine_interpolation not in {InterpolationMode.NEAREST, InterpolationMode.BILINEAR}:
        affine_interpolation = InterpolationMode.BILINEAR

    train_transform = v2.Compose([
        v2.ToImage(),
        ResizeLongestSide(
            image_size,
            interpolation=interpolation,
            antialias=True,
        ),
        PadToSquare(image_size, pad_fill),
        v2.RandomApply([
            v2.RandomAffine(
                degrees=4,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                interpolation=affine_interpolation,
                fill=pad_fill,
            )
        ], p=0.5),
        v2.RandomApply([v2.ColorJitter(brightness=0.10, contrast=0.10)], p=0.3),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    if weights is not None and exact_eval_transform:
        val_test_transform = weights.transforms()
    else:
        val_test_transform = v2.Compose([
            v2.ToImage(),
            ResizeLongestSide(
                image_size,
                interpolation=interpolation,
                antialias=True,
            ),
            PadToSquare(image_size, pad_fill),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

    return train_transform, val_test_transform

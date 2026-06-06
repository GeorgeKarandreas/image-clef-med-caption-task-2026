"""Image transform utilities for the caption task."""

from PIL import ImageOps

from utils.caption.helpers import preprocess_to_tensor

class SquarePad:
    """Pad a PIL image to square while keeping the original content centered."""

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image):
        width, height = image.size

        if width == height:
            return image

        max_side = max(width, height)

        pad_left = (max_side - width) // 2
        pad_right = max_side - width - pad_left

        pad_top = (max_side - height) // 2
        pad_bottom = max_side - height - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)

        return ImageOps.expand(image, padding, fill=self.fill)


class CaptionTestTransform:
    """Build both encoder inputs from one PIL image for TestImageDataset."""

    def __init__(self, biomed_preprocess, swin_preprocess, shared_transform=None):
        self.biomed_preprocess = biomed_preprocess
        self.swin_preprocess = swin_preprocess
        self.shared_transform = shared_transform

    def __call__(self, image):
        if self.shared_transform is not None:
            image = self.shared_transform(image)

        return {
            "biomed_pixels": preprocess_to_tensor(self.biomed_preprocess, image),
            "swin_pixels": preprocess_to_tensor(self.swin_preprocess, image),
        }

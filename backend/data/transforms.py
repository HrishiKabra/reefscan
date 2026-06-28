"""Preprocessing transforms for DINOv2-B. Phase 2.

Patch geometry (locked in CLAUDE.md, must be IDENTICAL in train and inference):
  - Training (current NOAA 2-class data): the dataset images ARE colony patches, so we use
    the WHOLE image — resize to 224x224, ImageNet-normalize. No point/centroid crop.
  - Inference: crop the SAM2 mask's bounding box, then resize 224 + normalize (same tail).
DINOv2 expects inputs sized to multiples of 14; 224 = 16 * 14.

`crop_patch_around_point` below is retained for FUTURE point-annotated sources (paired with
backend.data.split); it is NOT used by the current imagefolder pipeline.
"""
from __future__ import annotations

from PIL import Image
from torchvision import transforms

# DINOv2 uses ImageNet normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224

# Default square window (in source pixels) cropped around a point before resizing.
# CoralNet-style patch classification; tune in Phase 3 if needed.
DEFAULT_CROP_SIZE = 224


def build_transform(train: bool) -> transforms.Compose:
    """Return the tensor transform applied to an already-cropped PIL patch."""
    if train:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def crop_patch_around_point(img: Image.Image, row: int, col: int,
                            crop_size: int = DEFAULT_CROP_SIZE) -> Image.Image:
    """Crop a square `crop_size` window centered on (row, col), clamped to image bounds.

    (row, col) follow CoralNet convention: row = y (vertical), col = x (horizontal).
    """
    half = crop_size // 2
    width, height = img.size
    left = max(0, min(col - half, width - crop_size))
    top = max(0, min(row - half, height - crop_size))
    left = max(0, left)
    top = max(0, top)
    right = min(width, left + crop_size)
    bottom = min(height, top + crop_size)
    return img.crop((left, top, right, bottom))

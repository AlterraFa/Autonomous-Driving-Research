"""
Constants and type hints for image_transform module.
Exposes registered component and augment names to Pylance.
"""

from typing import Literal

ComponentName = Literal[
    "to_image",
    "crop",
    "resize",
    "to_dtype",
    "color_jitter",
    "gaussian_blur",
    "normalize",
    "horizontal_flip",
]

AugmentName = Literal[
    "augment",
    "normalization",
]

AVAILABLE_COMPONENTS: tuple[ComponentName, ...] = (
    "to_image",
    "crop",
    "resize",
    "to_dtype",
    "color_jitter",
    "gaussian_blur",
    "normalize",
    "horizontal_flip",
)

AVAILABLE_AUGMENTS: tuple[AugmentName, ...] = (
    "augment",
    "normalization",
)

__all__ = [
    "ComponentName",
    "AugmentName",
    "AVAILABLE_COMPONENTS",
    "AVAILABLE_AUGMENTS",
]

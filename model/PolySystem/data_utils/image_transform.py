import torch
import numpy as np
from torch import nn
from torchvision.transforms import functional as F
from torchvision.transforms import v2
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Type

from .constants import ComponentName, AugmentName
from .collator import MetaType

class GuidedCropV2(nn.Module):
    def __init__(self, top: float = 0.0, bottom: float = 1.0, left: float = 0.0, right: float = 1.0):
        super().__init__()
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def _get_dims(self, img):
        if isinstance(img, torch.Tensor):
            return img.shape[-2], img.shape[-1]
        else:
            return img.size[1], img.size[0]

        
    def forward(self, img):
        h, w = self._get_dims(img)
        
        top_crop   = int(h * self.top)
        bot_crop   = int(h * self.bottom)
        left_crop  = int(w * self.left)
        right_crop = int(w * self.right)
        
        crop_height = bot_crop - top_crop
        crop_width  = right_crop - left_crop
        
        return F.crop(img, top=top_crop, left=left_crop, height=crop_height, width=crop_width)

class Augment(nn.Module):
    def __init__(
        self, 
        dimension: tuple = (244, 244),
        crop: list = [0.0, 1.0],  # [top, bottom] or [top, bottom, left, right]
        color_jitter: float = 1.0, 
        color_distortion: bool = False, 
        gaussian_blur: bool = False, 
        normalization: tuple = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ):
        super().__init__()
        
        transforms_list = []

        transforms_list.append(GuidedCropV2(*crop))

        transforms_list.append(v2.Resize(dimension, antialias = True))

        transforms_list.append(v2.ToDtype(torch.float32, scale=True))
        
        if color_distortion:
            transforms_list.append(
                v2.RandomApply([
                    v2.ColorJitter(
                        brightness=0.8*color_jitter, 
                        contrast=0.8*color_jitter, 
                        saturation=0.8*color_jitter, 
                        hue=0.2*color_jitter
                    )
                ], p=0.6)
            )
            
        if gaussian_blur:
            transforms_list.append(
                v2.RandomApply([
                    v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                ], p=0.25)
            )
            
        transforms_list.append(
            v2.Normalize(mean=normalization[0], std=normalization[1])
        )
        
        self.aug = v2.Compose(transforms_list)

    def forward(self, x):
        with torch.no_grad():
            return self.aug(x)
    

class Normalization(nn.Module):
    def __init__(self, size, crop = None, normalization = None):
        super().__init__()
        transforms_list = []
        transforms_list.append(v2.ToImage()) 
        if crop is not None:
            transforms_list.append(GuidedCropV2(*crop))
        transforms_list.append(v2.Resize(size, antialias = True))
        transforms_list.append(v2.ToDtype(torch.float32, scale = True))
        if normalization is not None:
            transforms_list.append(v2.Normalize(mean=normalization[0], std=normalization[1]))
        self.aug = v2.Compose(transforms_list)

    def forward(self, x):
        return self.aug(x)

COMPONENT_REGISTRY: Dict[str, Type["BaseComponent"]] = {}
AUGMENT_REGISTRY: Dict[str, Type["BaseAugment"]] = {}

def register_augment(name: str):
    def decorate(cls: Type["BaseAugment"]):
        AUGMENT_REGISTRY.update({name: cls})
        return cls
    return decorate

def register_component(name: str):
    def decorate(cls: Type["BaseComponent"]):
        COMPONENT_REGISTRY.update({name: cls})
        return cls
    return decorate

def compose_augment(name: AugmentName, ctx: "AugmentContext", **kwargs):
    if name not in AUGMENT_REGISTRY:
        raise KeyError(f"Unknown augment: {name}. Available: {list(AUGMENT_REGISTRY.keys())}")
    return AUGMENT_REGISTRY[name](ctx=ctx, **kwargs)

def compose_component(name: ComponentName, ctx: "AugmentContext", **kwargs):
    if name not in COMPONENT_REGISTRY:
        raise KeyError(f"Unknown component: {name}. Available: {list(COMPONENT_REGISTRY.keys())}")
    return COMPONENT_REGISTRY[name](ctx=ctx, **kwargs)

@dataclass
class AugmentContext:
    size: Optional[Tuple[int, int]] = None
    crop: Optional[List[float]] = None
    normalization: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    color_jitter: float = 1.0
    color_distortion: bool = False
    gaussian_blur: bool = False
    antialias: bool = True
    to_image: bool = True
    dtype: torch.dtype = torch.float32
    scale: bool = True
    flip_prob: float = 0.0
    flip_meta_keys: List[str] = field(default_factory=lambda: [
        MetaType.Waypoint.value,
        MetaType.AuxWaypoint.value,
        MetaType.Steer.value,
        MetaType.TurnSignal.value,
        MetaType.Polyline.value,
    ])

class BaseAugment(nn.Module, ABC):
    def __init__(self, ctx: AugmentContext):
        super().__init__()
        self.ctx = ctx
        self.aug = self.build()

    @abstractmethod
    def build(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, x):
        return self.aug(x)

class BaseComponent(nn.Module, ABC):
    def __init__(self, ctx: AugmentContext):
        super().__init__()
        self.ctx = ctx

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def _split_input(self, x):
        if isinstance(x, tuple) and len(x) == 2:
            return x[0], x[1], True
        return x, None, False

    def _merge_input(self, images, meta, paired: bool):
        return (images, meta) if paired else images

    def _apply_to_images(self, images, fn):
        if isinstance(images, dict):
            return {k: fn(v) for k, v in images.items()}
        return fn(images)

    def _flip_xy(self, arr):
        if arr is None:
            return arr
        if isinstance(arr, (list, tuple)):
            flipped = [self._flip_xy(item) for item in arr]
            return type(arr)(flipped)
        if isinstance(arr, torch.Tensor):
            if arr.shape[-1] < 2:
                return arr
            out = arr.clone()
            out[..., 0] = -out[..., 0]
            return out
        if isinstance(arr, np.ndarray):
            if arr.shape[-1] < 2:
                return arr
            out = arr.copy()
            out[..., 0] = -out[..., 0]
            return out
        return arr

    def _flip_turn_signal(self, val):
        if isinstance(val, torch.Tensor):
            out = val.clone()
            out = torch.where(val == 1, torch.tensor(2, dtype=val.dtype, device=val.device), out)
            out = torch.where(val == 2, torch.tensor(1, dtype=val.dtype, device=val.device), out)
            return out
        if isinstance(val, np.ndarray):
            out = val.copy()
            out[val == 1] = 2
            out[val == 2] = 1
            return out
        if isinstance(val, str):
            low = val.lower()
            if "left" in low:
                return val.replace("left", "right")
            if "right" in low:
                return val.replace("right", "left")
        if isinstance(val, (list, tuple)):
            flipped = [self._flip_turn_signal(item) for item in val]
            return type(val)(flipped)
        return val

    def _flip_meta(self, meta):
        if meta is None:
            return meta
        if isinstance(meta, dict):
            out = dict(meta)
            for key, value in meta.items():
                if key in self.ctx.flip_meta_keys:
                    if key == "gt_data_steer":
                        out[key] = -value
                    elif key == "command_turn_signal":
                        out[key] = self._flip_turn_signal(value)
                    else:
                        out[key] = self._flip_xy(value)
                elif isinstance(value, dict):
                    out[key] = self._flip_meta(value)
            return out
        return meta


# ======================= Augmentation Components ============================ #
@register_component("to_image")
class ToImageComponent(BaseComponent):
    def forward(self, x):
        images, meta, paired = self._split_input(x)
        images = self._apply_to_images(images, v2.ToImage())
        return self._merge_input(images, meta, paired)

@register_component("crop")
class GuidedCropComponent(BaseComponent):
    def forward(self, x):
        images, meta, paired = self._split_input(x)
        if self.ctx.crop is None:
            return self._merge_input(images, meta, paired)
        images = self._apply_to_images(images, GuidedCropV2(*self.ctx.crop))
        return self._merge_input(images, meta, paired)

@register_component("resize")
class ResizeComponent(BaseComponent):
    def forward(self, x):
        images, meta, paired = self._split_input(x)
        if self.ctx.size is None:
            return self._merge_input(images, meta, paired)
        images = self._apply_to_images(images, v2.Resize(self.ctx.size, antialias=self.ctx.antialias))
        return self._merge_input(images, meta, paired)

@register_component("to_dtype")
class ToDtypeComponent(BaseComponent):
    def forward(self, x):
        images, meta, paired = self._split_input(x)
        images = self._apply_to_images(images, v2.ToDtype(self.ctx.dtype, scale=self.ctx.scale))
        return self._merge_input(images, meta, paired)

@register_component("color_jitter")
class ColorJitterComponent(BaseComponent):
    def forward(self, x):
        images, meta, paired = self._split_input(x)
        if not self.ctx.color_distortion:
            return self._merge_input(images, meta, paired)
        jitter = v2.RandomApply([
            v2.ColorJitter(
                brightness=0.8 * self.ctx.color_jitter,
                contrast=0.8 * self.ctx.color_jitter,
                saturation=0.8 * self.ctx.color_jitter,
                hue=0.2 * self.ctx.color_jitter,
            )
        ], p=0.6)
        images = self._apply_to_images(images, jitter)
        return self._merge_input(images, meta, paired)

@register_component("gaussian_blur")
class GaussianBlurComponent(BaseComponent):
    def forward(self, x):
        images, meta, paired = self._split_input(x)
        if not self.ctx.gaussian_blur:
            return self._merge_input(images, meta, paired)
        blur = v2.RandomApply([
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.25)
        images = self._apply_to_images(images, blur)
        return self._merge_input(images, meta, paired)

@register_component("normalize")
class NormalizeComponent(BaseComponent):
    def forward(self, x):
        images, meta, paired = self._split_input(x)
        if self.ctx.normalization is None:
            return self._merge_input(images, meta, paired)
        images = self._apply_to_images(
            images,
            v2.Normalize(mean=self.ctx.normalization[0], std=self.ctx.normalization[1])
        )
        return self._merge_input(images, meta, paired)
    
@register_component("horizontal_flip")
class HorizontalFlipComponent(BaseComponent):
    def forward(self, x):
        images, meta, paired = self._split_input(x)
        if self.ctx.flip_prob <= 0:
            return self._merge_input(images, meta, paired)

        # Get batch size (handle both batched and unbatched inputs)
        batch_size = self._get_batch_size(images)
        
        # Generate independent flip decision for each sample
        flip_mask = torch.rand(batch_size) < self.ctx.flip_prob
        
        # Apply flip only to samples where flip_mask is True
        if flip_mask.any():
            images = self._apply_flip_per_sample(images, flip_mask)
            meta = self._flip_meta_per_sample(meta, flip_mask) if meta else meta
        
        return self._merge_input(images, meta, paired)

    def _get_batch_size(self, images):
        """Extract batch size from images (handles dict and tensor)."""
        if isinstance(images, dict):
            first_img = next(iter(images.values()))
        else:
            first_img = images
        
        if isinstance(first_img, torch.Tensor):
            return first_img.shape[0]
        return len(first_img)

    def _apply_flip_per_sample(self, images, flip_mask):
        """Apply flip selectively per sample using vectorized ops."""
        def flip_fn(img):
            if img.dim() <= 2:
                return img  # Single image without batch dim
            
            # Vectorized: flip all samples where mask is True
            flipped_indices = torch.where(flip_mask)[0]
            if len(flipped_indices) == 0:
                return img
            
            flipped = img.clone()
            flipped[flipped_indices] = F.hflip(img[flipped_indices])
            return flipped
        
        return self._apply_to_images(images, flip_fn)

    def _flip_meta_per_sample(self, meta, flip_mask):
        """Apply metadata flip selectively per sample using mask."""
        if not isinstance(meta, dict):
            return meta
        
        out = dict(meta)
        for key, value in meta.items():
            if key in self.ctx.flip_meta_keys and value is not None:
                # Flip values where flip_mask is True
                flipped = self._flip_meta_key_per_sample(key, value, flip_mask)
                out[key] = flipped
            elif isinstance(value, dict):
                out[key] = self._flip_meta_per_sample(value, flip_mask)
        return out

    def _flip_meta_key_per_sample(self, key, value, flip_mask):
        """Flip specific metadata key type per sample using vectorized ops."""
        flipped_indices = torch.where(flip_mask)[0]
        if len(flipped_indices) == 0:
            return value
        
        if key == "gt_data_steer":
            # Vectorized steer negation
            if isinstance(value, torch.Tensor):
                out = value.clone()
                out[flipped_indices] = -value[flipped_indices]
                return out
            if isinstance(value, np.ndarray):
                out = value.copy()
                out[flipped_indices.cpu().numpy()] = -value[flipped_indices.cpu().numpy()]
                return out
        
        elif key == "command_turn_signal":
            # Turn signal flip (left ↔ right) - use set lookup to avoid iteration
            if isinstance(value, (list, tuple)):
                indices_set = set(flipped_indices.cpu().tolist())
                out = [self._flip_turn_signal(value[i]) if i in indices_set else value[i] 
                       for i in range(min(len(value), flip_mask.shape[0]))]
                return type(value)(out)
            else:
                # For tensor/array: stack flipped values at indices
                if isinstance(value, torch.Tensor):
                    out = value.clone()
                    if len(flipped_indices) > 0:
                        out[flipped_indices] = torch.stack([
                            self._flip_turn_signal(value[i]) for i in flipped_indices
                        ])
                    return out
                if isinstance(value, np.ndarray):
                    out = value.copy()
                    indices_np = flipped_indices.cpu().numpy() if isinstance(flipped_indices, torch.Tensor) else flipped_indices
                    if len(indices_np) > 0:
                        out[indices_np] = np.array([
                            self._flip_turn_signal(value[i]) for i in indices_np
                        ])
                    return out
        
        else:
            # Default: vectorized XY flip (waypoints, polyline, etc.)
            if isinstance(value, torch.Tensor):
                out = value.clone()
                if len(flipped_indices) > 0:
                    out[flipped_indices] = torch.stack([
                        self._flip_xy(value[i]) for i in flipped_indices
                    ])
                return out
            if isinstance(value, np.ndarray):
                out = value.copy()
                indices_np = flipped_indices.cpu().numpy() if isinstance(flipped_indices, torch.Tensor) else flipped_indices
                if len(indices_np) > 0:
                    out[indices_np] = np.array([
                        self._flip_xy(value[i]) for i in indices_np
                    ])
                return out
        
        return value

    
# ======================= Augmentation composer ============================ #
@register_augment("augment")
class ContextAugment(BaseAugment):
    def build(self) -> nn.Module:
        component_order = [
            "crop",
            "horizontal_flip",
            "resize",
            "to_dtype",
            "color_jitter",
            "gaussian_blur",
            "normalize",
        ]
        transforms_list: List[nn.Module] = [
            compose_component(name, ctx=self.ctx) for name in component_order
        ]
        return v2.Compose(transforms_list)

@register_augment("normalization")
class ContextNormalization(BaseAugment):
    def build(self) -> nn.Module:
        component_order = [
            "to_image",
            "crop",
            "resize",
            "to_dtype",
            "normalize",
        ]
        transforms_list: List[nn.Module] = [
            compose_component(name, ctx=self.ctx) for name in component_order
        ]
        return v2.Compose(transforms_list)
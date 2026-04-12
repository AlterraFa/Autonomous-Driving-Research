import torch
from torchvision.transforms import v2
from torch import nn
from torchvision.transforms import functional as F

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
    def __init__(self, size):
        super().__init__()
        transforms_list = []
        transforms_list.append(v2.ToImage()) 
        transforms_list.append(v2.Resize(size, antialias = True))
        transforms_list.append(v2.ToDtype(torch.float32, scale = True)) 
        self.aug = v2.Compose(transforms_list)

    def forward(self, x):
        return self.aug(x)
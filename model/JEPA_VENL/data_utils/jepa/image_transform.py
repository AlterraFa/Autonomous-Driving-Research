import sys
import torch
import numpy as np

from multiprocess import Value
from PIL import ImageFilter
from torchvision.transforms import functional as F

class DummyTensor:
    def __call__(self, pic):
        mode_to_nptype = {"I": np.int32, "I;16" if sys.byteorder == "little" else "I;16B": np.int16, "F": np.float32}
        img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
        return img

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))

class GuidedCrop:
    def __init__(self, top = 0.0, bottom = 1.0, left = 0.0, right = 1.0):
        self.top    = top
        self.bottom = bottom
        self.left   = left
        self.right  = right
        
    def __call__(self, img):
        w, h = img.size
        top_crop   = int(h * self.top)
        bot_crop   = int(h * self.bottom)
        left_crop  = int(w * self.left)
        right_crop = int(w * self.right)
        return F.crop(img, top = top_crop, left = left_crop, height = bot_crop - top_crop, width = right_crop - left_crop)

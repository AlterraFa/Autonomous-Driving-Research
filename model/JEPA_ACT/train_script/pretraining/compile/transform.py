from typing import Optional
from ....augmenter.transforms_builder import VideoTransform

def make_transforms(
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3 / 4, 4 / 3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    pad_frame_count: Optional[int] = None,
    pad_frame_method: str = "circulant",
):
    _frames_augmentation = VideoTransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
        normalize=normalize,
        pad_frame_count=pad_frame_count,
        pad_frame_method=pad_frame_method,
    )
    return _frames_augmentation

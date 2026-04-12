import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.data._utils.collate import collate
from torch.nn.utils.rnn import pad_sequence

from enum import Enum
from typing import List
from utils.messages.logger import Logger

class ImageType(Enum):
    I0 = "I0"
    MU = "MU"
    MR = "MR"
    Mask = "Mask"
    
class MetaType(Enum):
    Waypoint = "gt_data_midlane_wp"
    AuxWaypoint = "gt_data_aux_wp"
    Steer = "gt_data_steer"
    Throttle = "gt_data_throttle"
    Velocity = "gt_data_velocity"
    TurnSignal = "command_turn_signal"
    Polyline = "command_polycmd"
    GPS = "condition_GPS"
    Heading = "condition_heading"
    RoadType = "condition_road_type"
    Timestamp = "timestamp"

_image_members = set(ImageType)
_meta_members  = set(MetaType)

class RoadCollator:
    __slots__ = ['pop_img', 'pop_meta', 'transform', 'logger', 'pad_values']
    
    def __init__(self, image_types: List[ImageType] = None, meta_types: List[MetaType] = None, transform = None, pad_values = 0):
        self.logger = Logger()
        self.pad_values = pad_values

        if image_types is None:
            input_img_members = all_img_members
        elif not isinstance(image_types, (list, tuple, set)):
            input_img_members = {image_types}
        else:
            input_img_members = set(image_types)

        pop_img_members = _image_members.difference(input_img_members)
        self.pop_img = {member.value for member in pop_img_members}


        if meta_types is None:
            input_meta_members = all_meta_members
        elif not isinstance(meta_types, (list, tuple, set)):
            input_meta_members = {meta_types}
        else:
            input_meta_members = set(meta_types)
            
        pop_meta_members = _meta_members.difference(input_meta_members)
        self.pop_meta = {member.value for member in pop_meta_members}
        
        self.transform = transform
            
        input_img_keys_names = {member.name for member in input_img_members}
        pop_img_keys_names = {member.name for member in pop_img_members}
        
        input_meta_keys_names = {member.name for member in input_meta_members}
        pop_meta_keys_names = {member.name for member in pop_meta_members}
            
        self.logger.INFO(f"Images to KEEP: {', '.join(input_img_keys_names)}. Images to POP: {', '.join(pop_img_keys_names)}")
        self.logger.INFO(f"Meta to KEEP: {', '.join(input_meta_keys_names)}. Meta to POP: {', '.join(pop_meta_keys_names)}")
        
        
    def __call__(self, batch):
        images, metadatas = zip(*batch)

        batched_images = default_collate(images)
        if self.pop_img:
            for itype in self.pop_img:
                batched_images.pop(itype)
        
        batched_metas  = self._meta_collate(metadatas)
        if self.pop_meta:
            for mtype in self.pop_meta:
                batched_metas.pop(mtype)

        if self.transform:
            self.transform((batched_images, batched_metas))
                
        return batched_images, batched_metas


    def _meta_collate(self, metadatas):
        elem = metadatas[0]

        if isinstance(elem, dict):
            return {key: self._meta_collate([d[key] for d in metadatas]) for key in elem}
        
        elif isinstance(elem, np.ndarray):
            is_uniform_shape = all(x.shape == elem.shape for x in metadatas)
            if is_uniform_shape:
                return default_collate(metadatas)
            else:
                return pad_sequence([torch.as_tensor(x) for x in metadatas], batch_first = True, padding_value = self.pad_values)
            
        else:
            return default_collate(metadatas)
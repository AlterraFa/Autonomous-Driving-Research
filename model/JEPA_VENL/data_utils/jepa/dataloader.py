import os, sys
script_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torchvision
import torch

from torch.utils.data import random_split

class JEPALoader(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform = None, target_transform = None, fraction = 1):
        super().__init__(root, transform, target_transform)
        
        if fraction != 1.0:
            temp_sample = []
            sample_size = len(self.samples)
            for idx, (img, class_id) in enumerate(self.samples):
                temp_sample += [(img, class_id)]
                if idx >= sample_size * fraction: break
            self.samples = temp_sample
    
    def split(self, train = 0.9, val = 0.1):
        
        n_total = self.__len__()
        n_train = int(n_total * train)
        n_val   = int(n_total * val)
        n_test  = n_total - n_train - n_val

        return random_split(self, [n_train, n_val, n_test])
        


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from model.JEPA_VENL.data_utils.jepa.multiblock import JEPACollator
    from model.JEPA_VENL.data_utils.jepa.compose import transform_composer
    from model.JEPA_VENL.data_utils.jepa.image_transform import DummyTensor
    
    ROOT_DIR    = os.path.dirname(__file__)
    DATASET_DIR = os.path.join(os.path.dirname(ROOT_DIR), "../../data/JEPA")

    composer = transform_composer()
    dataset  = JEPALoader(DATASET_DIR, DummyTensor(), fraction = 0.1)
    collator = JEPACollator(nenc = 1, npred = 4)
    dataloader = DataLoader(dataset, 64, False, num_workers = 8, persistent_workers = True)
    
    total_sum = torch.zeros(3, dtype = torch.float)
    total_sq  = torch.zeros(3, dtype = torch.float)
    total_pixel = 0
    for img, _ in dataloader:
        img = img / 255.0
        total_sum += img.sum((0, 1, 2))
        total_sq  += (img ** 2).sum((0, 1, 2))

        total_pixel = img.shape[0] * img.shape[1] * img.shape[2]
        
    mean = total_sum / total_pixel
    dev  = (total_sq / total_pixel) - mean ** 2
    std_dev = dev ** .5
        
    print("Image standard dev color:", std_dev)
    print("Image mean color:", mean)
import os, sys

script_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
import pandas as pd
import numpy as np
import glob
import torch
import re
from PIL import Image
from turbojpeg import TurboJPEG
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import random_split


jpeg_loader = TurboJPEG("/usr/lib/libturbojpeg.so.0")

def decode_image(path):
    # -- All of this is RGB
    try:
        if path.lower().endswith(('.jpg', '.jpeg')):
            with open(path, "rb") as f:
                img_bytes = f.read()

            return jpeg_loader.decode(img_bytes, pixel_format = 0)
        else: 
            with Image.open(path) as img:
                return np.array(img.convert('RGB'))

    except Exception as e:
        print(f"Error processing {path=}: {e}")
        return None
    
def decode_batch_image(paths):
    with ThreadPoolExecutor(max_workers = 8) as executor:
        frames = np.asarray(list(executor.map(decode_image, paths)), dtype = np.uint8)
    return frames

class VideoDataset(Dataset):
    def __init__(
        self,
        data_paths, # -- CSVs Contains videos path
        dataset_fpc = None, 
        frames_per_clips = 16,
        nclips = 1,
        frame_step = 1,
        shared_transform = None,
        individual_transform = None,
        allow_clip_overlap = False,  
        random_jiggle_part = True,
        random_part = True,
    ):
        super().__init__()

        self.data_paths = data_paths
        self.def_fpc = frames_per_clips
        self.nclips  = nclips
        self.frame_step = frame_step
        self.allow_clip_overlap = allow_clip_overlap
        self.datasets_fpc = dataset_fpc
        self.random_jiggle_part = random_jiggle_part
        self.random_part = random_part
        self.individual_transform = individual_transform
        self.shared_transform = shared_transform

        
        if self.datasets_fpc is None:
            self.datasets_fpc = [frames_per_clips for _ in range(len(data_paths))]
        samples, labels = [], []
        self.nsamples_per_dataset = []
        for data_path in self.data_paths:
            df = pd.read_csv(data_path, header = None, delimiter = ",")            
            samples += list(df.values[:, 1])
            labels  += list(df.values[:, 2])
            nsamples = len(df)
            self.nsamples_per_dataset += [nsamples]
            
        self.video_indicies_map = []
        for idx, nsamples in enumerate(self.nsamples_per_dataset):
            self.video_indicies_map += [idx for _ in range(nsamples)]
            
        self.samples = samples
        self.labels  = labels

        
    def __getitem__(self, index):
        RETRIES = 5; retry = 0
        while True:
            sample = self.load_image_sequences(index)
            if sample is not None: break
            elif retry < RETRIES:
                print(f"Something went wrong at {index=}, retrying.")
                retry += 1
            else: return None
            
        return sample

    def load_image_sequences(self, index):
        sample = self.samples[index]
        dataset_idx = self.video_indicies_map[index]
        fpc = self.datasets_fpc[dataset_idx]
        
        buffer, clip_indicies = self._load_sequences(sample, fpc)
        loaded_video = len(buffer) > 0
        if not loaded_video: return None
        
        
        if self.shared_transform is not None:
            self.shared_transform(buffer)
            
        buffer = [buffer[(idx - 1) * len(indicies): idx * len(indicies)] for idx, indicies in enumerate(clip_indicies, start = 1)]
        if self.individual_transform is not None:
            buffer = [self.individual_transform(clip) for clip in buffer]
            
        return buffer, clip_indicies
        
        
        
    def _load_sequences(self, sample, fpc):
        image_seq_paths = glob.glob(os.path.join(sample, "*"))
        image_seq_paths = sorted(image_seq_paths, key = lambda x: re.findall(r'\d+', x.rsplit('.', 1)[0])[-1])
        seq_length = len(image_seq_paths)
        
        fstp = self.frame_step
        target_len = int(fstp * fpc)
        part_len = len(image_seq_paths) // self.nclips
        
        
        
        buffer_indices, clip_indices = [], []
        for i in range(self.nclips):
            if part_len > target_len:
                end_idx = target_len
                if self.random_jiggle_part:
                    end_idx = np.random.randint(target_len, part_len)
                start_idx = end_idx - target_len
                
                local_indicies = np.linspace(start_idx, end_idx, fpc)
                local_indicies = np.clip(local_indicies, start_idx, end_idx - 1).astype(np.int64)
                
                global_indices = local_indicies + i * part_len
            else:
                if not self.allow_clip_overlap:
                    local_indicies = np.linspace(0, part_len, num = part_len // fstp)
                    local_indicies = np.concatenate(
                        [
                            local_indicies,
                            np.ones(fpc - part_len // fstp) * part_len
                        ]
                    )
                    local_indicies = np.clip(local_indicies, 0, part_len - 1).astype(np.int64)

                    global_indices = local_indicies + i * part_len
                else:
                    sample_length = min(target_len, seq_length)
                    local_indicies = np.linspace(0, sample_length, num = sample_length // fstp)
                    local_indicies = np.clip(local_indicies, 0, sample_length - 1).astype(np.int64)
                    
                    if seq_length < target_len:
                        step = 0
                    else: step = (seq_length - target_len) // (self.nclips - 1)
                        
                    global_indices = local_indicies + i * step
            clip_indices += [global_indices.tolist()]
            buffer_indices.extend(global_indices.tolist())     
            
        buffer = decode_batch_image(np.array(image_seq_paths)[buffer_indices])
        return buffer, clip_indices
    
    def __len__(self):
        return len(self.samples)

    def split(self, train = 0.9, val = 0.1):
        
        n_total = self.__len__()
        n_train = int(n_total * train)
        n_val   = int(n_total * val)
        n_test  = n_total - n_train - n_val

        return random_split(self, [n_train, n_val, n_test])
                    

if __name__ == "__main__":
    import yaml
    from model.JEPA_ACT.augmenter.transforms_builder import VideoTransform
    from model.JEPA_ACT.masks.multiseq_multiblock3d import MaskCollator
    from torch.utils.data import DataLoader

    
    check_norm_val  = False
    check_collator  = True
    check_transform = True
    check_dataloader = False
    
    with open("./JEPA_ACT/cfgs/pretrain-224px-512.12e-384.12p.yaml", "r") as f:
        args = yaml.safe_load(f)

    mask_cfg     = args["mask"]
    dataset_fpcs = [16, 16]
    crop_size    = 224 
    patch_size   = 16 
    tubelet_size = 2
    
    if check_transform:
        transforms = VideoTransform(
            random_horizontal_flip = True,
            random_resize_scale = (0.8, 1.0),
            reprob = 0,
            motion_shift = True,
            crop_size = crop_size,
            normalize = ((0.2809, 0.2959, 0.2946), (0.2469, 0.2675, 0.2795))
        )
    if check_collator:
        collator = MaskCollator(
            cfgs_mask = mask_cfg,
            dataset_fpcs = dataset_fpcs,
            crop_size = (crop_size, crop_size),
            patch_size = (patch_size, patch_size),
            tubelet_size = tubelet_size
        )
    
    dataset = VideoDataset(
        [
            "./JEPA_ACT/data_bdd100k_val.csv",
        ],
        frame_step = 4,
        nclips = 1,
        allow_clip_overlap = True,
        individual_transform = transforms if check_transform else None,
        dataset_fpc = dataset_fpcs
    )

    if check_collator:
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 16,
            shuffle = True,
            collate_fn = collator if check_collator else None,    
        )
        
        # -- [[(sub_B, H, W, buff_length, 3), (sub_B, ), (sub_B, buff_length)], [(sub_B, num_enc1), (sub_B, num_enc2), ...], [(sub_B, num_pred1), (sub_B, num_pred2), ...]] * nclips
        batch = next(iter(dataloader))
        print("Number of fpcs:", len(batch)) # -- Number of dataset_fpcs
        print("Number of metadatas:", len(batch[0])) # -- Image, encoder_indicies, mask_indicies
        print("Number of cfg masks", len(batch[0][1])) # -- number of cfg masks
        print("Image shape:", batch[0][0][0][0].shape)
        
        for i in range(len(batch)):
            for enc_mask in batch[i][1]:
                print(enc_mask.shape) # -- 
                
    if check_dataloader:
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 1,
            shuffle = True,
            collate_fn = collator if check_collator else None,    
        )
    
        loader = iter(dataloader)
        for idx in range(len(dataloader)):
            print(len(dataloader))
            next(loader)
            print(idx)

    if check_norm_val:
        device = torch.device("cuda")
        
        total_sum = torch.zeros(3, dtype = torch.float64).to(device)
        total_sq  = torch.zeros(3, dtype = torch.float64).to(device)
        total_pixel = 0
        for idx in range(len(dataset)):
            buffer, _ = dataset[idx]
            buffer = torch.tensor(np.array(buffer), dtype = torch.float64).to(device)
            buffer /= 255.0
            total_sum += buffer.sum((0, 1, 2, 3), dtype = torch.float64)
            total_sq  += (buffer ** 2).sum((0, 1, 2, 3), dtype = torch.float64)

            total_pixel += torch.prod(torch.tensor(buffer.shape[:-1]))


        mean = total_sum / total_pixel
        dev  = (total_sq / total_pixel) - mean ** 2
        std_dev = torch.sqrt(dev)
        
        print(f"{mean=}") # -- [0.2809, 0.2959, 0.2946]
        print(f"{std_dev=}") # -- [0.2469, 0.2675, 0.2795]
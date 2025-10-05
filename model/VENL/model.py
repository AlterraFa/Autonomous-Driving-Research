import cv2
import torch
import queue
import threading
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings


from torch import optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader


from tqdm.auto import tqdm
from pathlib import Path
from utils.messages.logger import Logger
from typing import Literal, Optional, Union

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

class VENL(nn.Module):
    def __init__(self, droprate: float = 0.1):
        self.log = Logger()
        super().__init__()

        # Might change to WRN to improve performance
        self.multicam_backbone: nn.ModuleList[nn.Sequential] = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(in_channels = 3, out_channels = 24, kernel_size = 5, stride = 2),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 5, stride = 2),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Conv2d(in_channels = 36, out_channels = 48, kernel_size = 3, stride = 2),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Conv2d(in_channels = 48, out_channels = 64, kernel_size = 3, stride = 1),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Flatten()
            ]) for _ in range(3)
        ])

        # Shallow network to prevent translational + rotational invariance
        self.unrouted_backbone: nn.Sequential = nn.Sequential(*[
            nn.Conv2d(in_channels = 1, out_channels = 24, kernel_size = 5, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),

            nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 5, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),

            nn.Conv2d(in_channels = 36, out_channels = 48, kernel_size = 3, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),

            nn.Flatten()
        ])

        self.routed_backbone: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 24, kernel_size = 5, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),

            nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 5, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),

            nn.Flatten()
        )

        self.feature_downsize = nn.Sequential(
            nn.LazyLinear(1000),
            nn.ReLU(),
            nn.Dropout(droprate),

            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(droprate),
        )

        self.fusion_projector = nn.Sequential(
            nn.LazyLinear(100),
            nn.LeakyReLU(),
            nn.Dropout(droprate)
        )
    
        self.initialized = False

    def _init_weights(self):
        """Custom weight initialization for all submodules."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["log"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.log = Logger()

    @classmethod
    def steer(cls, camera_shape = (80, 200), map_shape = (50, 50), components: int = 3, droprate = 0.1) -> "VENL":
        self = cls(droprate = droprate)
        self.components = components
        self.log.INFO("Using steer mode")

        self.gmm_head = nn.Linear(100, 3 * components) # 3 gaussian parameters * number of modes
        self.determ_head = nn.Linear(100, 1)
        self.input_metadata = {
            "I0": (1, 3, *camera_shape),
            "I1": (1, 3, *camera_shape),
            "I2": (1, 3, *camera_shape),
            "MU": (1, 1, *map_shape),
            "MR": (1, 3, *map_shape),
        }
        self.output_names = ["steer", "weights", "muy", "sigma"]
        
        return self

    @classmethod
    def waypoint(cls, camera_shape = (80, 200), map_shape = (50, 50), num_waypoints = 1, components: int = 3, droprate = 0.1) -> "VENL":
        self = cls(droprate = droprate)
        self.num_waypoints = num_waypoints
        self.components = components
        self.log.INFO("Using waypoint mode")

        self.gmm_head = nn.Linear(100, components * (1 + num_waypoints * 4)) # 1 weights, num_waypoints * 2 mean, num_waypoints * 2 standard deviation
        self.determ_head = nn.Linear(100, num_waypoints * 2)
        self.input_metadata = {
            "I0": (1, 3, *camera_shape),
            "I1": (1, 3, *camera_shape),
            "I2": (1, 3, *camera_shape),
            "MU": (1, 1, *map_shape),
            "MR": (1, 3, *map_shape),
        }
        self.output_names = ["waypoint", "weights", "muy", "sigma"]

        return self

    def initialize_module(self, I0: torch.Tensor, I1: torch.Tensor, I2: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor):
        if self.initialized == False:
            self.initialized = True

            self.forward(I0, I1, I2, MU, MR)
            self._init_weights()
            self.log.INFO("Layer initialized")
        else:
            self.log.WARNING("Layer already initialized")
        
        

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, I2: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor) -> torch.Tensor:
        argcount = self.forward.__code__.co_argcount
        argnames = self.forward.__code__.co_varnames[: argcount]

        if self.initialized == False:
            self.log.ERROR(f"Modules not initialized", exit_code = -1)
        
        if not torch.onnx.is_in_onnx_export():
            for name in argnames[1: ]: # skip self
                tensor = locals()[name]
                expected_shape = self.input_metadata.get(name)
                if expected_shape[1:] != tuple(tensor.shape)[1:]:
                    self.log.ERROR(f"Input tensor {name} has shape {tensor.shape[1:]}, expected {expected_shape[1:]}", exit_code = 12)

        # features of multicam setup
        f0 = self.multicam_backbone[0](I0)
        f1 = self.multicam_backbone[1](I1)
        f2 = self.multicam_backbone[2](I2)
        # features of unrouted map
        fmu = self.unrouted_backbone(MU)

        # Concatenation of left, front, right and map features on a single vector
        features_cat = torch.cat([f0, f1, f2, fmu], dim=1) # TENSORRT DOES NOT SUPPORT HSTACK OR VSTACK

        out = self.feature_downsize(features_cat)
        routed_features = self.routed_backbone(MR)

        gmm_out = self.gmm_head(out)
        determ_in = torch.cat([out, routed_features], dim = 1)
        determ_out = self.determ_head(self.fusion_projector(determ_in))

        if self.output_names[0] == 'waypoint':
            return determ_out.view(-1, self.num_waypoints, 2), *self.extract_gparams(gmm_out)
        else:
            return determ_out, *self.extract_gparams(gmm_out)

    def extract_gparams(self, gmm_params: torch.Tensor):
        if not hasattr(self, "num_waypoints"):
            # predetermined 3 parameters correspond to 3 chunks 
            weights, muy_weights, sigma_weights = torch.chunk(gmm_params, 3, 1)
            weights = torch.softmax(weights, dim=1) 
            muy     = muy_weights                       
            sigma   = torch.exp(sigma_weights)
            return weights, muy, sigma
        else:
            weights, muy_weights, sigma_weights = torch.split(
                gmm_params, 
                [
                    self.components, 
                    self.components * self.num_waypoints * 2, 
                    self.components * self.num_waypoints * 2
                ],  # 1 weights, num_waypoints * 2 mean, num_waypoints * 2 standard deviation per components
                dim=1
            )
            weights = torch.softmax(weights, dim=1).unsqueeze(-1)
            muy     = muy_weights.view(-1, self.components, self.num_waypoints, 2)
            sigma   = torch.exp(sigma_weights).view(-1, self.components, self.num_waypoints, 2)  # (batch, modes, waypoints, dim)
            return weights, muy, sigma

    def postprocess(self, data):
        return self.extract_gparams(data)

    def gaussian_function(self, sample, parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        weights, muy, sigma = parameters
        if not hasattr(self, "num_waypoints"):
            try:
                probs_per_components = self._univariate(sample, muy, sigma)
                return weights * probs_per_components  # Return GMM probability per mode with weights
            except Exception as e:
                self.log.ERROR(
                    f"Please check the parameters if it is in the format of univariate or if the sample format is correct. Sample: [bold]{sample.shape}[/], Paramters: [bold]{muy.shape}[/]",
                    full_traceback = e,
                    exit_code = 12

                )
        else:
            if muy.shape[-2] != sample.shape[1]:
                self.log.ERROR(f"Mismatch number of waypoints. Sample: [bold]{sample.shape}[/], Paramters: [bold]{muy.shape}[/]", exit_code = 12)
            try:
                probs_per_components = self._multivariate(sample, muy, sigma)
                return weights * probs_per_components  # returns joint probability of x, y per mode per waypoint
            except Exception as e:
                self.log.ERROR(
                    f"Please check the parameters if it is in the format of multivariate or if the sample format is correct. Sample: [bold]{sample.shape}[/], Parameters: [bold]{muy.shape}[/]",
                    full_traceback = e,
                    exit_code = 12
                )


    @staticmethod
    def _univariate(sample, muy, sigma):
        return (1 / (2 * torch.pi * sigma ** 2) ** 0.5) * torch.exp(-(sample - muy) ** 2 / (2 * sigma ** 2))


    @staticmethod
    def _multivariate(sample, muy, sigma):
        """Format for sample must be (B, wp, 2)"""
        _, N, *_ = muy.shape
        sample = sample.unsqueeze(1).expand(-1, N, -1, -1)

        # joint probability distribution between x and y => norm const is prod while exp term is sum
        norm_const = (1.0 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * sigma)).prod(dim=3)
        exp_term = torch.exp(-0.5 * (((sample - muy) / sigma) ** 2).sum(dim=3))
        return norm_const * exp_term

def single_epoch_training(model: VENL, loader: DataLoader, optimizer: optim, target_spread: Union[torch.Tensor, float], l1_weight: float, l2_weight: float, l1 = 0.0, l2 = 0.0):
    model.train()
    device = next(model.parameters()).device

    trainBar = tqdm(loader, desc = "Train", position = 1, leave = False)
    trainMetrics = {"Total": 0, "MSE": 0, "NLL": 0, "STD Reg": 0}
    for images, true_waypoints, controls, _ in trainBar:
        optimizer.zero_grad(set_to_none = True)

        images = {name: image.to(device) for name, image in images.items()}
        gt     = true_waypoints.to(device) if model.output_names[0] == "waypoint" else controls['steer'].unsqueeze(1).to(device)

        determ, weights, muy, sigma = model(**images)        
        gmm_prob_per_mode = model.gaussian_function(gt, (weights, muy, sigma))
        total_gmm_prob    = gmm_prob_per_mode.sum(1)

        # Uncertainty loss
        nll_loss   = -torch.log(total_gmm_prob + 1e-11).mean()
        # Deterministic loss
        mse_loss   = F.mse_loss(determ, gt)
        # Discourage unnecessary components
        l1_weights_norm = weights.abs().mean()
        l2_weights_norm = weights.pow(2.0).mean()
        # encourages uncertainty more (smaller std_dev penalizes harder than the big ones)
        # each component and x, y will have the same target std_dev
        std_reg    = F.mse_loss(torch.log(sigma), target_spread.view(1, 1, -1, 1).to(device))
        # Model weight reg
        weightParams = [p for n, p in model.named_parameters()
                        if p.requires_grad and "weight" in n]
        l1Norm = sum(p.abs().mean() for p in weightParams)
        l2Norm = sum(p.pow(2.0).mean() for p in weightParams)

        loss = nll_loss + \
            mse_loss + \
            l1_weights_norm * l1_weight + \
            l2_weights_norm * l2_weight + \
            std_reg + \
            l1Norm * l1 + \
            l2Norm * l2
        
        loss.backward()
        optimizer.step()

        trainMetrics["Total"]  += loss.item()
        trainMetrics["MSE"]    += mse_loss.item()
        trainMetrics["NLL"]    += nll_loss.item()
        trainMetrics["STD Reg"] += std_reg.item()

        avg_total = trainMetrics["Total"] / (trainBar.n + 1)
        avg_mse   = trainMetrics["MSE"] / (trainBar.n + 1)
        avg_nll   = trainMetrics["NLL"] / (trainBar.n + 1)
        avg_std   = trainMetrics["STD Reg"] / (trainBar.n + 1)

        trainBar.set_postfix({
            "Total": f"{avg_total:.4f}",
            "MSE": f"{avg_mse:.4f}",
            "NLL": f"{avg_nll:.4f}",
            "STD": f"{avg_std:.4f}",
        })


    trainMetrics["Total"]   /= len(loader)
    trainMetrics["MSE"]     /= len(loader)
    trainMetrics["NLL"]     /= len(loader)
    trainMetrics["STD Reg"] /= len(loader)
    
    del images, true_waypoints, gt, determ, weights, muy, sigma
    torch.cuda.empty_cache()

    return trainMetrics

def single_epoch_val(model: VENL, loader: DataLoader, target_spread: torch.Tensor, l1_weight: float, l2_weight: float):
    model.eval()
    device = next(model.parameters()).device

    valBar = tqdm(loader, desc = "Val", position = 2, leave = False)
    valMetrics = {"Total": 0, "MSE": 0, "NLL": 0, "STD Reg": 0}
    with torch.no_grad():
        for images, true_waypoints, controls, _ in valBar:

            images = {name: image.to(device) for name, image in images.items()}
            gt     = true_waypoints.to(device) if model.output_names[0] == "waypoint" else controls['steer'].unsqueeze(1).to(device)

            determ, weights, muy, sigma = model(**images)        
            gmm_prob_per_mode = model.gaussian_function(gt, (weights, muy, sigma))
            total_gmm_prob    = gmm_prob_per_mode.sum(1)

            # Uncertainty loss
            nll_loss   = -torch.log(total_gmm_prob + 1e-11).mean()
            # Deterministic loss
            mse_loss   = F.mse_loss(determ, gt)
            # Discourage unnecessary components
            l1_weights_norm = weights.abs().mean()
            l2_weights_norm = weights.pow(2.0).mean()
            # encourages uncertainty more (smaller std_dev penalizes harder than the big ones)
            # each component and x, y will have the same target std_dev
            std_reg    = F.mse_loss(torch.log(sigma), target_spread.view(1, 1, -1, 1).to(device))

            loss = nll_loss + \
                mse_loss + \
                l1_weights_norm * l1_weight + \
                l2_weights_norm * l2_weight + \
                std_reg

            valMetrics["Total"]  += loss.item()
            valMetrics["MSE"]    += mse_loss.item()
            valMetrics["NLL"]    += nll_loss.item()
            valMetrics["STD Reg"] += std_reg.item()

            avg_total = valMetrics["Total"] / (valBar.n + 1)
            avg_mse   = valMetrics["MSE"] / (valBar.n + 1)
            avg_nll   = valMetrics["NLL"] / (valBar.n + 1)
            avg_std   = valMetrics["STD Reg"] / (valBar.n + 1)

            valBar.set_postfix({
                "Total": f"{avg_total:.4f}",
                "MSE": f"{avg_mse:.4f}",
                "NLL": f"{avg_nll:.4f}",
                "STD": f"{avg_std:.4f}",
            })

    valMetrics["Total"]   /= len(loader)
    valMetrics["MSE"]     /= len(loader)
    valMetrics["NLL"]     /= len(loader)
    valMetrics["STD Reg"] /= len(loader)

    del images, true_waypoints, gt, determ, weights, muy, sigma
    torch.cuda.empty_cache()

    return valMetrics

class CarlaDatasetLoader:
    def __init__(self, dataset_dir: str, images_key: list[str] = None, downsize_ratio = 1, load_size: int = -1, shuffle = True):
        self.log = Logger()
        self.dataset_dir = Path(dataset_dir)
        self.img_dir     = self.dataset_dir / "images"
        self.meta_dir    = self.dataset_dir / "metadata"
        self.images_key  = images_key
        
        if not self.img_dir.exists() or not self.meta_dir.exists():
            raise FileNotFoundError(f"Dataset directories not found: expected 'images/' and 'metadata/'.")
        
        self.samples_dir = [f_name for f_name in self.meta_dir.glob("*.npy")]
        self.samples_dir = np.array(self.samples_dir)
        num_samples      = len(self.samples_dir)
        if load_size != -1 and load_size != len(self.samples_dir):
            if shuffle:
                rand_idx = np.random.randint(0, len(self.samples_dir), load_size)
                self.samples_dir = self.samples_dir[rand_idx]
            else: 
                self.samples_dir = self.samples_dir[np.arange(0, load_size, 1)]

        self.log.INFO(f"Found {num_samples} samples in {self.dataset_dir}. Using {len(self.samples_dir)} samples")

        self.loader = AsyncLoader()
        self.downsize_ratio = downsize_ratio

    def __len__(self):
        return len(self.samples_dir)
    
    def read_image(self, img_path: Path):
        """Reads image as RGB or grayscale, ensures shape (..., 1 or 3)."""
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Handle grayscale automatically → (H, W, 1)
        if len(image.shape) == 2:
            image = image[..., None]  # Add channel dimension

        # Convert BGR → RGB only if it's color
        elif image.shape[2] == 3:
            image = image[:, :, ::-1]

        # Downsize if requested
        if self.downsize_ratio != 1:
            H, W = image.shape[:2]
            image = cv2.resize(image, (W // self.downsize_ratio, H // self.downsize_ratio))
            if image.ndim == 2:  # Resize can drop channel dim
                image = image[..., None]

        return image

    def _get_samples(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index out of range")

        meta_file = Path(self.samples_dir[idx])
        meta      = np.load(meta_file, allow_pickle = True).item()

        if self.images_key is None:
            self.log.WARNING("Single input detected", once = True)
            img_file  = self.dataset_dir / meta["img_file"]
            self.loader.load(self.read_image, str(img_file))
            image     = self.loader.get_result(True)[:, :, ::-1]
            if self.downsize_ratio != 1:
                H, W, _   = image.shape
                image = cv2.resize(image, (W // self.downsize_ratio, H // self.downsize_ratio))

            return {
                "image": image,
                "ego_waypoints": np.array(meta["ego_waypoints"], dtype=np.float32),
                "control": meta["control"],
                "turn_signal": meta["turn_signal"],
                "timestamp": meta["timestamp"],
            }
        else:
            self.log.WARNING("Multiple inputs detected", once = True)
            inp_images = {}
            for key_name in self.images_key:
                img_file  = self.dataset_dir / meta["img_file"][key_name]
                self.loader.load(self.read_image, str(img_file))
                image     = self.loader.get_result(True)[:, :, ::-1] # Inverted color channel
                if self.downsize_ratio != 1:
                    H, W, _   = image.shape
                    image = cv2.resize(image, (W // self.downsize_ratio, H // self.downsize_ratio))
                inp_images.update({key_name: image})

            return {
                "images": inp_images,
                "ego_waypoints": np.array(meta["ego_waypoints"], dtype=np.float32),
                "control": meta["control"],
                "turn_signal": meta["turn_signal"],
                "timestamp": meta["timestamp"],
            }
    
    def __getitem__(self, idx):
        return self._get_samples(idx)

    CONTROL_KEYS = ["steer", "throttle", "brake", "velocity"]

    @classmethod
    def collate_fn(cls, batch):
        """
        Collate function supporting both single- and multi-input image cases.
        - Single input: data["image"] -> (H, W, C)
        - Multi input:  data["image"] -> dict[str, np.ndarray]
        """

        first_item = batch[0]["images"]

        # --- Single-input case ---
        if isinstance(first_item, np.ndarray):
            images = torch.stack([
                torch.from_numpy(np.ascontiguousarray(data["images"])) for data in batch
            ]).permute(0, 3, 1, 2) / 255.0 # Normalize 

        # --- Multi-input case ---
        elif isinstance(first_item, dict):
            images = {}
            input_keys = list(first_item.keys())

            for key_name in input_keys:
                imgs = [
                    torch.from_numpy(np.ascontiguousarray(data["images"][key_name])) for data in batch
                ]
                imgs = torch.stack(imgs).permute(0, 3, 1, 2) / 255.0
                images[key_name] = imgs
        else:
            raise TypeError(f"Unsupported image type in batch: {type(first_item)}")

        # --- Non-image metadata ---
        wp = torch.stack([torch.from_numpy(data["ego_waypoints"]) for data in batch])

        controls = {
            key: torch.tensor(
                [data["control"][key] for data in batch],
                dtype=torch.float32
            )
            for key in cls.CONTROL_KEYS
        }

        turn_signals = torch.tensor(
            [data["turn_signal"] for data in batch], dtype=torch.long
        )

        return images, wp, controls, turn_signals

    def split(self, train = 0.9, val = 0.1):
        
        n_total = self.__len__()
        n_train = int(n_total * train)
        n_val   = int(n_total * val)
        n_test  = n_total - n_train - n_val

        return random_split(self, [n_train, n_val, n_test])


class AsyncLoader:
    def __init__(self):
        self.q = queue.Queue()         # tasks
        self.results = queue.Queue()   # results
        self.running = True
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        while self.running:
            try:
                func, args, kwargs = self.q.get(timeout=1)
                result = func(*args, **kwargs)
                self.results.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                self.results.put(e)

    def load(self, func, *args, **kwargs):
        """Enqueue a function call. Returns immediately."""
        self.q.put((func, args, kwargs))

    def get_result(self, block=True, timeout=None):
        """
        Retrieve the next available result.
        If no result is ready:
          - block=True waits until available (or timeout).
          - block=False returns immediately (or raises queue.Empty).
        """
        return self.results.get(block=block, timeout=timeout)

    def stop(self):
        self.running = False
        self.worker.join()
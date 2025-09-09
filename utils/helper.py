import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch import randperm
from torchvision.transforms  import functional as TF

from typing import Tuple, List, Optional, Callable, Union, Sequence

class EarlyStopping:
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.0, 
                 path: str = "checkpoint.pt",
                 verbose: bool = False):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.verbose   = verbose
        self.counter   = 0
        self.best_loss = torch.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module):
        # check if loss improved by at least min_delta
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model, self.path)
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}. Saved model to {self.path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in val loss for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                

class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor = None, transform=None):
        if labels is not None:
            assert images.shape[0] == labels.shape[0] 
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]        # shape [C,H,W]
        if self.labels is not None:
            y = self.labels[idx]        # one‑hot float
        if self.transform:
            x = self.transform(x)
        return x, y if self.labels is not None else x
    
class UnlabeledDataset(Dataset):
    def __init__(self, images: torch.Tensor, weak_transform=None, strong_transform=None):
        self.images = images
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]  # shape [C, H, W]
        x_weak = self.weak_transform(x) if self.weak_transform else x
        x_strong = self.strong_transform(x) if self.strong_transform else x
        return x_weak, x_strong

def valSplit(train: tuple, split: float = 0.1):
    N = train[0].size(0)
    valLength   = int(split * train[0].shape[0])
    trainLength = train[0].shape[0] - valLength
    perm        = randperm(N, generator=torch.Generator().manual_seed(42))
    train_idx   = perm[:trainLength]
    val_idx     = perm[trainLength:]
    
    return (train[0][train_idx], train[1][train_idx]), (train[0][val_idx], train[1][val_idx])


class VariableTensorDataset(Dataset):
    def __init__(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        augments: Optional[List[Callable]] = None,
        release: Optional[Union[int, Sequence[int]]] = None,
    ):
        """
        images:   (N, C, H, W) tensor
        labels:   (N, ...) tensor or None
        augments: list of callables that map a Tensor->[Tensor]
        release:  which augment indices to return in __getitem__;
                  if None, uses self.aug_idx (single view)
                  if int or [i,j,...], returns tuple of views
        """
        super().__init__()
        if labels is not None:
            assert labels.shape[0] == images.shape[0], (
                f"Images/labels length mismatch: {images.shape[0]} vs {labels.shape[0]}"
            )

        assert isinstance(augments, (list, type(None))), "augments must be a list or None"
        self.images   = images
        self.labels   = labels
        self.augments = augments or []
        # the “default” single-view index
        self.aug_idx  = 0

        # normalize release into a list of ints, or None
        if release is None:
            self.release = None
        else:
            if isinstance(release, int):
                self.release = [release]
            else:
                # assume sequence of ints
                self.release = list(release)
            # sanity-check
            for i in self.release:
                assert 0 <= i < len(self.augments), f"release index {i} out of range"

    def set_augment(self, idx: int):
        """Change the single-view augment index (used when release is None)."""
        assert 0 <= idx < len(self.augments), "augment index out of range"
        self.aug_idx = idx

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, index):
        x = self.images[index]

        if self.augments:
            if self.release is not None:
                views = [self.augments[i](x) for i in self.release]
                x = tuple(views)
            else:
                x = self.augments[self.aug_idx](x)

        if self.labels is not None:
            return x, self.labels[index]
        else:
            return x, index
        
        
class VariableThresh(nn.Module):
    def __init__(self, unlabeledSz: int, numClasses: int, tau = .8):
        super().__init__()
        
        learningTracer = torch.full((unlabeledSz, ), -1, dtype = torch.long)
        self.register_buffer("learningTracer", learningTracer)
        
        classArray = torch.arange(numClasses).unsqueeze(0)
        self.register_buffer("classArray", classArray)
        
        self.N = unlabeledSz
        self.numClasses = numClasses
        self.tau = tau
        
    def forward(self, prob: torch.Tensor, indices):
        maxClass           = torch.argmax(prob, dim = 1, keepdim = True).expand(-1, self.numClasses)
        confs, pseudoLabel = prob.max(dim = 1)
        mask               = (confs >= self.tau)
        learnEffect        = (mask.unsqueeze(1) * (self.classArray == maxClass)).sum(dim = 0)
        
        print(torch.all((self.classArray == maxClass) == (pseudoLabel.unsqueeze(1) == self.classArray)))

        confidentIndicies  = indices.to(self.learningTracer.device)[mask]
        self.learningTracer[confidentIndicies] = pseudoLabel[mask]
        
        
        unused = (self.learningTracer == -1).sum()
        if torch.max(learnEffect) < unused:
            denom = torch.max(torch.stack([
                torch.max(learnEffect),
                (self.N - torch.sum(learnEffect)).clone().detach()
            ]))
            beta = learnEffect / torch.clamp(denom, min = 1.0) # Per class (ratio of original threshold, also determines if the model is confident in that class)
        else:
            beta = learnEffect / torch.clamp(torch.max(learnEffect), min = 1.0) # Per class
            
        beta = self.project(beta)
            
        variMask = confs >= (beta[pseudoLabel] * self.tau)
        return variMask, pseudoLabel

    project = lambda self, x: x / (2 + x)

normAugment = T.Compose([
    T.Lambda(lambda x: x.float().div(255)),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

noAugment = T.Compose([
])

weakAugment = T.Compose([
    T.RandomApply([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=(96, 96), padding=8, pad_if_needed=True),
        T.ToTensor()
    ], p=0.5),       
])

strongAugment = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops = 3, magnitude = 10),
    T.ToTensor()
])

def allSetAugment(
    train: Tuple[torch.Tensor, torch.Tensor],
    test: Tuple[torch.Tensor, torch.Tensor],
    unlabeled: torch.Tensor,
    batchSize: int,
    muy: float,
    splitRatio: float = 0.1
):
    train_images, train_labels = train
    num_train = train_images.shape[0]
    split = int(num_train * splitRatio)

    # split off a validation slice
    trainDS = TensorImageDataset(
        train_images[split:], 
        train_labels[split:].long(),
        transform=weakAugment
    )
    valDS   = TensorImageDataset(
        train_images[:split],
        train_labels[:split].long(),
        transform=None
    )
    testDS  = TensorImageDataset(
        test[0],
        test[1].long(),
        transform=None
    )
    unlabeledDS = TensorImageDataset(
        unlabeled
    )

    trainLoader     = DataLoader(trainDS, batch_size=batchSize, shuffle=True)
    valLoader       = DataLoader(valDS,   batch_size=batchSize, shuffle=True)
    testLoader      = DataLoader(testDS,  batch_size=batchSize, shuffle=False)
    unlabeledLoader = DataLoader(
        unlabeledDS, 
        batch_size=int(muy * batchSize), 
        shuffle=True
    )

    return trainLoader, valLoader, testLoader, unlabeledLoader

class DistributionAlignment(nn.Module):
    def __init__(self, labels: torch.tensor, numClasses: int, momentum: float):
        super(DistributionAlignment, self).__init__()
        
        counts = torch.bincount(labels)
        pEmperical = (counts.float() / labels.numel())
        
        self.register_buffer("pEmperical", pEmperical)
        self.register_buffer("pRunning", torch.zeros(numClasses))
        self.momentum = momentum
        
    def forward(self, q: torch.Tensor):
        pBatch = q.mean(dim = 0)
        self.pRunning = (
            self.momentum * self.pRunning 
            + (1 - self.momentum) * pBatch
        )
        
        labelTilde = q * (self.pEmperical / (self.pRunning + 1e-6)).unsqueeze(0)
        
        return labelTilde / labelTilde.sum(dim = 1, keepdim = True)

        

class DetectionDataset(Dataset):
    def __init__(self, dataset, className: List[str], imgSize = 640):
        super().__init__()
        
        self.dataset = dataset
        self.imgSize = imgSize
        self.CLASSES = className
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, target = self.dataset[index]
        ann = target['annotation']
        objs = ann['object']
        if isinstance(objs, dict):
            objs = [objs]

        boxes = torch.tensor([
            [
            float(o['bndbox']['xmin']),
            float(o['bndbox']['ymin']),
            float(o['bndbox']['xmax']),
            float(o['bndbox']['ymax'])
            ]
        for o in objs], dtype = torch.float32)
        labels = torch.tensor([self.CLASSES.index(o['name']) for o in objs], dtype = torch.long)
        
        H, W = img.shape[1:]
        scale = min(self.imgSize / H, self.imgSize / W)
        newH, newW = int(H * scale), int(W * scale)
        img = TF.resize(img, (newH, newW))
        padH, padW = self.imgSize - newH, self.imgSize - newW
        left = padW // 2
        right = padW - left
        top = padH // 2
        bottom = padH - top
        img = TF.pad(img, (left, top, right, bottom), fill = 0)
        
        boxes = boxes.clone().float()
        boxes *= scale
        boxes[:, [0, 2]] += left
        boxes[:, [1, 3]] += top
        boxes /= self.imgSize
        
        return img, (labels, boxes)
    
    def collate_fn(self, batch):
        imgs      = [item[0] for item in batch]
        labels    = [item[1][0] for item in batch]
        boxes     = [item[1][1] for item in batch]

        imgs = torch.stack(imgs, dim=0)
        return imgs, labels, boxes

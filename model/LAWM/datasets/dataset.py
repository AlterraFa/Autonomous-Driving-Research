import torch
import numpy as np
from torch.utils.data import Sampler

class DistributedWeightedSampler(Sampler):
    """Weighted sampling with replacement that is sharded across distributed ranks."""

    def __init__(self, weights, num_samples, num_replicas=1, rank=0, seed=0):
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas - 1}], got {rank}")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}")
        if self.weights.numel() == 0:
            raise ValueError("weights must be non-empty")
        if float(self.weights.sum().item()) <= 0.0:
            raise ValueError("weights sum must be > 0")

        self.total_size = int(np.ceil(self.num_samples / self.num_replicas)) * self.num_replicas
        self.num_samples_per_rank = self.total_size // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=g,
        ).tolist()

        rank_indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples_per_rank


class WeightedSampler(Sampler):
    """Non-distributed weighted sampling with replacement and epoch control."""

    def __init__(self, weights, num_samples, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}")
        if self.weights.numel() == 0:
            raise ValueError("weights must be non-empty")
        if float(self.weights.sum().item()) <= 0.0:
            raise ValueError("weights sum must be > 0")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=g,
        ).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples
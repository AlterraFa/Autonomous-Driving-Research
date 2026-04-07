# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch
import torch.distributed as dist

from .logger import Logger

logger = Logger(__name__)

def init_distributed(port=37129, rank_and_world_size=(None, None)):
    # try to set all environment variables to avoid triggering a segfault
    # environment variables can be reallocated during the execution of torch.distributed.init_process_group
    # the idea is a race condition may trigger if init_progress_group is modifying an environment variable at
    # the same time as Python, so we try to set all environs before initializing distributed
    if "SLURM_JOB_ID" in os.environ:
        # Use the slurm_tmpdir (if it exists) instead of /tmp
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ["MASTER_ADDR"] = "localhost"

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        except Exception:
            logger.INFO("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank

    if int(world_size) <= 1:
        return 1, int(rank)

    try:
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(
            backend="nccl", 
            world_size=world_size, 
            rank=rank, 
            device_id=torch.device(f"cuda:{rank}")
        )
    except Exception as e:
        world_size, rank = 1, 0
        logger.ERROR(f"Rank: {rank}. Distributed training not available", full_traceback = e)
        

    return world_size, rank

class DifferentiableDistGather(torch.autograd.Function):
    """
    Gathers tensors from all GPUs while preserving the backward gradient path.
    Standard dist.all_gather breaks the graph; this class fixes it.
    """
    @staticmethod
    def forward(ctx, tensor):
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        
        # Create storage for gathered tensors
        gathered = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor)
        
        # Save the size for the backward pass
        ctx.batch_size = tensor.size(0)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grads = grad_output.chunk(ctx.world_size, dim=0)
        return grads[ctx.rank]

def all_gather(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    return DifferentiableDistGather.apply(tensor)

class DifferentiableDistAllReduce(torch.autograd.Function):
    """
    Performs an All-Reduce (Average) while preserving the autograd graph.
    Standard dist.all_reduce is in-place and breaks the graph.
    """
    @staticmethod
    def forward(ctx, tensor):
        ctx.world_size = dist.get_world_size()
        reduced_tensor = tensor.clone()
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
        return reduced_tensor / ctx.world_size

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def all_reduce(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    return DifferentiableDistAllReduce.apply(tensor)
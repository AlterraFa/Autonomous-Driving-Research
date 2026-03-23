import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random
import time
import math
from typing import List, Union
from abc import ABC, abstractmethod


class GradientOptim(ABC):
    def __init__(self, optimizer, reduction = 'mean'):
        self._reduction = reduction
        self._optimizer = optimizer
        self._task_grads = {} # Stores {param_obj: [task1_grad, task2_grad...]}
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self._task_grads[p] = []
                p.register_hook(self._make_hook(p))
    
    @property
    def optimizer(self):
        return self._optimizer
        
    def zero_grad(self):
        for p in self._task_grads:
            self._task_grads[p] = []
        return self._optimizer.zero_grad()

    def _make_hook(self, p):
        """Closure Factory to create parameter extraction hooks"""
        def hook(grad):
            self._task_grads[p].append(grad.clone())
            return grad
        return hook

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def backward(self, *objectives):
        pass
        
    
    @property
    def clear_grad(self):
        for p in self._task_grads.keys():
            self._task_grads[p] = []
            
            
class NormalGrad(GradientOptim):
    def __init__(self, optimizer):
        super().__init__(optimizer)

    def backward(self, *objectives):
        if self._reduction == "mean":
            total_loss = sum(objectives) / len(objectives)
        elif self._reduction == "sum":
            total_loss = sum(objectives)
        else: raise NotImplemented("Your reduction method does not exist")
        total_loss.backward()

    def step(self):
        self.optimizer.step()
        self.clear_grad # Clean up hooks

class PCGrad(GradientOptim):
    def step(self):
        for p, grads in self._task_grads.items():
            if not grads: continue
            
            if len(grads) > 1:
                clean_grad = self._surgery(grads)
            else:
                clean_grad = grads[0]
            
            p.grad = clean_grad
            
        self.optimizer.step()
        self.clear_grad
    
    def backward(self, *objectives):
        num_obj = len(objectives)
        for idx, obj in enumerate(objectives):
            self.optimizer.zero_grad()
            obj.backward(retain_graph = idx < (num_obj - 1))

    def _surgery(self, grads: list[torch.Tensor]):
        shape = grads[0].shape
        num_elem = grads[0].numel()
        grads = [g.flatten() for g in grads]
        pc_grad = copy.deepcopy(grads)
        for idx, gradient in enumerate(pc_grad):
            compare_grads = [g for i, g in enumerate(grads) if i != idx]
            random.shuffle(compare_grads)
            for comp_grad in compare_grads:
                scalar_prod = torch.dot(gradient, comp_grad)
                if scalar_prod < 0: # -- Detected 2 Vector with angle > 90
                    gradient -= scalar_prod * (comp_grad / (comp_grad.norm() ** 2))
                   
        grad = torch.concat(pc_grad, dim = 0).reshape(-1, num_elem) 
        if self._reduction == "mean":
            grad = grad.mean(0).reshape(shape)
            return grad
        elif self._reduction == "sum":
            grad = grad.sum(0).reshape(shape)
            return grad

class GradNorm(GradientOptim):
    def __init__(self, optimizer, n_tasks, alpha=1.5, w_lr=0.025, device = 'cuda'):
        super().__init__(optimizer)
        
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.l0 = None
        self.w = nn.Parameter(torch.ones(n_tasks, device = device))
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr)
        
        self.task_idx = 0
        
    def _make_hook(self, p):
        def hook(grad):
            self._task_grads[p].append((self.task_idx, grad))
            return torch.zeros_like(grad)
        return hook

    def backward(self, *objectives):

        if self.l0 is None:
            self.l0 = torch.tensor([obj.item() for obj in objectives], device = objectives[0].device)
        
        num_obj = len(objectives)
        for idx in range(num_obj):
            self.task_idx = idx
            self.optimizer.zero_grad()
            objectives[idx].backward(retain_graph = True)

        self._register_shared_layer()
        shared_grad = [g for i, g in self._task_grads[self.last_shared_layers]]
        self._train_norm(shared_grad, objectives) 
        


    def _register_shared_layer(self):
        """Find the final shared layer between each objectives"""
        if not hasattr(self, 'last_shared_layers'):
            for p, grads in self._task_grads.items():
                if len(grads) != self.n_tasks:
                    break
                self.last_shared_layers = p
            
        if not hasattr(self, 'last_shared_layers'):
            raise RuntimeError(f"No shared parameters detected at {self.n_tasks} given objectives")


    def _train_norm(self, shared_grad, objectives):
        """Compute the weighting parameter"""
        task_norms = torch.stack([
            self.w[idx] * torch.norm(grad, p = 2) 
            for idx, grad in enumerate(shared_grad)
        ]) 
        
        with torch.no_grad():
            mean_grad = torch.mean(task_norms)

            loss_ratio = torch.tensor([obj.item() for obj in objectives], device = self.w.device) / self.l0
            inv_lr = loss_ratio / torch.mean(loss_ratio)

            tgt_norm = mean_grad * (inv_lr ** self.alpha)
            
        grad_norm_loss = (task_norms - tgt_norm).abs().mean()
        self.w_opt.zero_grad()
        grad_norm_loss.backward()

    def step(self):
        with torch.no_grad():
            for p, grad_info in self._task_grads.items():
                if p.grad is not None:
                    p.grad.zero_()
                else:
                    p.grad = torch.zeros_like(p)
                    
                # -- Dynamic weighted loss
                for task_idx, grad_tensor in grad_info:
                    p.grad.add_(self.w[task_idx].detach() * grad_tensor)

        self.w_opt.step()
        with torch.no_grad():
            normalize_coeff = self.n_tasks / self.w.sum()
            self.w.data.mul_(normalize_coeff)

        self.optimizer.step()
        self.clear_grad

class FAMO(GradientOptim):
    def __init__(self, optimizer, n_tasks, gamma = 0.01, w_lr = 0.025, max_norm = 1.0, device = 'cuda', reduction='mean'):
        super().__init__(optimizer, reduction)
        self.w = torch.tensor([0.0] * n_tasks, requires_grad=True, device = device)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        
        self.min_losses = None
        self.curr_loss = None
        self.prev_loss = None
    
    def backward(
        self, 
        *objectives, 
        ):
        if len(objectives) > 1:
            objectives = torch.stack(objectives)
        else:
            objectives = objectives[0]
        self._set_min_loss(objectives)
        
        self.curr_loss = objectives.detach().clone()
        
        loss = self._get_weighted_loss(objectives)        
        loss.backward()
        
    def step(self):
        if self.prev_loss is not None:
            delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                         (self.curr_loss - self.min_losses + 1e-8).log()
            with torch.enable_grad():
                z = F.softmax(self.w, dim = -1)
                d_w = torch.autograd.grad(z, self.w, grad_outputs = delta)[0]

            self.w_opt.zero_grad()
            self.w.grad = d_w
            self.w_opt.step()

        if self.max_norm > 0:
            params = [p for group in self.optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params, self.max_norm)
        
        self.optimizer.step()
        self.prev_loss = self.curr_loss
        self.clear_grad

    def _set_min_loss(self, obs):
        if self.min_losses is None:
            self.min_losses = obs.detach().clone()
        else:
            self.min_losses = torch.min(self.min_losses, obs.detach())

    
    def _get_weighted_loss(self, objectives: torch.Tensor):
        z = F.softmax(self.w, dim = -1)
        D = objectives - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss
        

# --- Gradient Optimizer Registry ---
GRADIENT_OPTIMIZER_REGISTRY = {
    "normal": NormalGrad,
    "pcgrad": PCGrad,
    "gradnorm": GradNorm,
    "famo": FAMO,
}


def create_gradient_optimizer(
    optimizer_name: str,
    base_optimizer: torch.optim.Optimizer,
    n_tasks: int = None,
    **kwargs
) -> GradientOptim:
    """
    Factory function to create a gradient optimizer from registry.
    
    Args:
        optimizer_name: Name of optimizer ('normal', 'pcgrad', 'gradnorm', 'famo')
        base_optimizer: PyTorch optimizer instance
        n_tasks: Number of tasks (required for GradNorm and FAMO)
        **kwargs: Additional arguments specific to the optimizer
                 - For GradNorm: alpha, w_lr, device
                 - For FAMO: gamma, w_lr, max_norm, device
                 - For PCGrad: reduction
                 - For NormalGrad: (none)
    
    Returns:
        Initialized GradientOptim instance
    """
    optimizer_name = optimizer_name.lower().strip()
    
    if optimizer_name not in GRADIENT_OPTIMIZER_REGISTRY:
        available = ", ".join(GRADIENT_OPTIMIZER_REGISTRY.keys())
        raise ValueError(
            f"Unknown gradient optimizer: '{optimizer_name}'. "
            f"Available options: {available}"
        )
    
    optimizer_class = GRADIENT_OPTIMIZER_REGISTRY[optimizer_name]
    
    # Build arguments based on optimizer type
    if optimizer_name == "normal":
        return optimizer_class(base_optimizer)
    
    elif optimizer_name == "pcgrad":
        reduction = kwargs.get("reduction", "mean")
        return optimizer_class(base_optimizer, reduction=reduction)
    
    elif optimizer_name == "gradnorm":
        if n_tasks is None:
            raise ValueError("n_tasks is required for GradNorm optimizer")
        alpha = kwargs.get("alpha", 1.5)
        w_lr = kwargs.get("w_lr", 0.025)
        device = kwargs.get("device", "cuda")
        return optimizer_class(
            base_optimizer,
            n_tasks=n_tasks,
            alpha=alpha,
            w_lr=w_lr,
            device=device
        )
    
    elif optimizer_name == "famo":
        if n_tasks is None:
            raise ValueError("n_tasks is required for FAMO optimizer")
        gamma = kwargs.get("gamma", 0.01)
        w_lr = kwargs.get("w_lr", 0.025)
        max_norm = kwargs.get("max_norm", 1.0)
        device = kwargs.get("device", "cuda")
        reduction = kwargs.get("reduction", "mean")
        return optimizer_class(
            base_optimizer,
            n_tasks=n_tasks,
            gamma=gamma,
            w_lr=w_lr,
            max_norm=max_norm,
            device=device,
            reduction=reduction
        )
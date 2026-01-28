import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any

from model.ImprovedVENL.impl.model import ImprovedVENL

# ============================================================================
# LOSS REGISTRY SYSTEM
# ============================================================================

LOSS_REGISTRY: Dict[str, type] = {}

def register_loss(name: str):
    """Decorator to register a loss class."""
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


class BaseLoss(nn.Module, ABC):
    """Base class for all loss components."""
    
    def __init__(self, ctx: "LossContext"):
        """Initialize with shared context."""
        super().__init__()
        self.ctx = ctx
    
    @abstractmethod
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        """Compute loss. Returns scalar loss tensor."""
        pass
    
    def __call__(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        return self.forward(pred, gt_wp, aux_wp)


class LossContext:
    """Shared context for all loss modules."""
    
    def __init__(
        self,
        device: torch.device,
        wp_coeff: torch.Tensor,
        target_sep: torch.Tensor,
        target_std: torch.Tensor,
        gaussian_fn,
        components: int,
        pad_value: float,
        model_params,
        loss_coeffs
    ):
        self.device = device
        self.wp_coeff = wp_coeff
        self.target_sep = target_sep
        self.target_std = target_std
        self.gaussian_fn = gaussian_fn
        self.components = components
        self.pad_value = pad_value
        self.model_params = model_params
        self.loss_coeffs = loss_coeffs


# ============================================================================
# WAYPOINT DISTANCE LOSSES
# ============================================================================

@register_loss("soft_dist")
class SoftDistLoss(BaseLoss):
    """Soft MSE loss using log(cosh) for robustness."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        pred_wp = pred[0]
        diff = pred_wp - gt_wp
        abs_diff = torch.abs(diff)
        loss = abs_diff + torch.nn.functional.softplus(-2.0 * abs_diff) - torch.log(torch.tensor(2.0))
        
        weights = self.ctx.wp_coeff.view(1, -1, 1) # [1, Num_Waypoints, 1]
        return (loss * weights).sum((1, 2)).mean()


@register_loss("mse")
class MseLoss(BaseLoss):
    """Standard MSE loss with weighted waypoints."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        pred_wp = pred[0]
        element_wise_loss = F.mse_loss(pred_wp, gt_wp, reduction="none")
        weights = self.ctx.wp_coeff.view(1, -1, 1)
        return (element_wise_loss * weights).sum((1, 2)).mean()


@register_loss("dir")
class DirectionLoss(BaseLoss):
    """Cosine similarity loss between predicted and ground truth direction vectors."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        pred_wp = pred[0]
        
        pred_dir = pred_wp[:, 1:] - pred_wp[:, :-1]
        gt_dir = gt_wp[:, 1:] - gt_wp[:, :-1]
        
        pred_dir = F.normalize(pred_dir, p=2, dim=-1)
        gt_dir = F.normalize(gt_dir, p=2, dim=-1)
        
        cos_sim = (pred_dir * gt_dir).sum(dim=-1)
        return (1.0 - cos_sim).mean()


@register_loss("smoothness")
class SmoothnnessLoss(BaseLoss):
    """Encourages smooth trajectories by penalizing acceleration and jerk."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        pred_wp = pred[0]
        
        v = pred_wp[:, 1:] - pred_wp[:, :-1]
        a = v[:, 1:] - v[:, :-1]
        j = a[:, 1:] - a[:, :-1]
        
        return a.pow(2).mean() + j.pow(2).mean()


@register_loss("lat_lon")
class LatLonLoss(BaseLoss):
    """Separate lateral and longitudinal error along trajectory direction."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        pred_wp = pred[0]
        
        tangent = F.normalize(gt_wp[:, 1:] - gt_wp[:, :-1], p=2, dim=-1)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        
        error = pred_wp[:, 1:] - gt_wp[:, 1:]
        
        lat_err = (error * normal).sum(dim=-1).abs().mean()
        lon_err = (error * tangent).sum(dim=-1).abs().mean()
        
        return lat_err + lon_err


# ============================================================================
# GMM LIKELIHOOD & ASSIGNMENT LOSSES
# ============================================================================

@register_loss("nll")
class GmmNllLoss(BaseLoss):
    """Gaussian Mixture Model negative log-likelihood loss."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        _, weights, muy, sigma = pred
        
        # Compute gaussian probability per mode
        weights_exp = torch.exp(weights)
        gmm_prob_per_mode = self.ctx.gaussian_fn(sample=aux_wp, parameters=(weights_exp, muy, sigma))
        
        # Mask out invalid probabilities
        mask_aux = aux_wp.sum((-1, -2), keepdim=True) != (torch.prod(torch.tensor(aux_wp.shape[2:])) * self.ctx.pad_value)
        masked_gmm_prob = gmm_prob_per_mode * mask_aux
        
        # Compute total probability and NLL
        mask_sum = mask_aux.sum(1)
        safe_denominator = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum * 2)
        total_gmm_prob = (masked_gmm_prob.sum(1) / safe_denominator).sum(1)
        total_gmm_prob_clamped = torch.clamp(total_gmm_prob, min=1e-6)
        nll_loss = (-torch.log(total_gmm_prob_clamped)).mean()
        
        return nll_loss


# ============================================================================
# REGULARIZATION LOSSES
# ============================================================================

@register_loss("std_reg")
class StdRegLoss(BaseLoss):
    """Regularize GMM sigma to match target spread."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        _, _, _, sigma = pred
        
        t_std_view = torch.exp(self.ctx.target_std).view(1, 1, -1, 1).expand_as(sigma)
        std_reg = F.mse_loss(sigma, t_std_view)
        return std_reg


@register_loss("entropy")
class EntropyLoss(BaseLoss):
    """Entropy regularization to encourage confident component selection."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        _, weights, _, _ = pred
        
        probs = torch.exp(weights)
        entropy_w = -(probs * weights).sum(dim=1).mean()
        return entropy_w


@register_loss("l1_gmm")
class L1GmmLoss(BaseLoss):
    """L1 regularization on GMM mixing weights."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        _, weights, _, _ = pred
        return weights.abs().mean()


@register_loss("l2_gmm")
class L2GmmLoss(BaseLoss):
    """L2 regularization on GMM mixing weights."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        _, weights, _, _ = pred
        return weights.pow(2.0).mean()


@register_loss("l1_model")
class L1ModelLoss(BaseLoss):
    """L1 regularization on model weights."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        weight_params = [p for n, p in self.ctx.model_params if p.requires_grad and "weight" in n]
        if not weight_params:
            return torch.tensor(0.0, device=self.ctx.device, requires_grad=True)
        return sum(p.abs().mean() for p in weight_params)


@register_loss("l2_model")
class L2ModelLoss(BaseLoss):
    """L2 regularization on model weights."""
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> torch.Tensor:
        weight_params = [p for n, p in self.ctx.model_params if p.requires_grad and "weight" in n]
        if not weight_params:
            return torch.tensor(0.0, device=self.ctx.device, requires_grad=True)
        return sum(p.pow(2.0).mean() for p in weight_params)

@register_loss("soft_dtw")
class SoftDTW(BaseLoss):
    def forward(self, pred, gt_wp, aux_wp):
        pred_wp = pred[0]
        dist_matrix = torch.cdist(pred_wp, gt_wp, p = 2) ** 2
        B, N, M = dist_matrix.shape
        
        gamma = self.ctx.loss_coeffs['lda']
        D = torch.full((B, N + 1, M + 1), float('inf'), device=dist_matrix.device)
        D[:, 0, 0] = 0 
        
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                upper = D[:, i - 1, j]
                left  = D[:, i, j - 1]
                diag  = D[:, i - 1, j - 1]
                                
                stacked = torch.stack([-upper/gamma, -left/gamma, -diag/gamma], dim=1)
                soft_min = -gamma * torch.logsumexp(stacked, dim=1)
                
                D[:, i, j] = dist_matrix[:, i - 1, j - 1] + soft_min
        
        loss = D[:, N, M]
        
        return loss.mean()


class NavLoss(nn.Module):
    """
    Modular loss orchestrator that composes multiple pluggable loss components.
    
    Usage:
        loss_fn = NavLoss(
            nav_model=model,
            device=device,
            wp_coeff=[1.0, 1.0, 0.5, ...],
            target_sep=[...],
            target_std=[...],
            enabled_losses=["soft_dist", "nll", "std_reg", "entropy", "l2_model"],
            loss_coeffs={"soft_dist": 1.0, "nll": 1.0, ...},
            pad_value=-1.0,
            delta=0.25,
        )
        total_loss, logs = loss_fn(pred, gt_wp, aux_wp)
    """
    
    def __init__(
        self,
        nav_model: ImprovedVENL,
        device: torch.device,
        wp_coeff: list,
        target_sep: list,
        target_std: list,
        enabled_losses: list = None,
        loss_coeffs: dict = None,
        pad_value: float = -1.0,
        delta: float = 0.25,
    ):
        super().__init__()
        
        self.model = nav_model
        self.device = device
        self.pad_value = pad_value
        self.delta = delta
        
        # Convert to tensors
        self.target_sep = torch.as_tensor(target_sep, dtype=torch.float32).to(device)
        self.target_std = torch.as_tensor(target_std, dtype=torch.float32).to(device)
        self.wp_coeff = torch.as_tensor(wp_coeff, dtype=torch.float32).to(device)
        
        # Create shared context for all loss modules
        self.ctx = LossContext(
            device=device,
            wp_coeff=self.wp_coeff,
            target_sep=self.target_sep,
            target_std=self.target_std,
            gaussian_fn=self.model.gaussian_function,
            components=self.model.components,
            pad_value=pad_value,
            model_params=list(self.model.named_parameters()),
            loss_coeffs = loss_coeffs
        )
        
        # Default loss coefficients
        default_coeffs = {
            "soft_dist": 1.0,
            "mse": 0.0,
            "nll": 1.0,
            "std_reg": 1.0,
            "entropy": 0.0,
            "l1_gmm": 0.0,
            "l2_gmm": 0.0,
            "l1_model": 0.0,
            "l2_model": 0.0001,
            "dir": 0.0,
            "smoothness": 0.0,
            "lat_lon": 0.0,
            "lda": 0.001
        }
        
        # Merge with provided coefficients
        self.loss_coeffs = {**default_coeffs, **(loss_coeffs or {})}
        
        # Default enabled losses (only those with non-zero coefficients)
        if enabled_losses is None:
            enabled_losses = [k for k, v in self.loss_coeffs.items() if v > 0.0]
        
        self.enabled_losses = enabled_losses
        
        # Initialize loss modules
        self.loss_modules = nn.ModuleDict()
        for loss_name in self.enabled_losses:
            if loss_name not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss '{loss_name}'. Available: {list(LOSS_REGISTRY.keys())}")
            loss_class = LOSS_REGISTRY[loss_name]
            self.loss_modules[loss_name] = loss_class(self.ctx)
    
    def forward(self, pred: List[torch.Tensor], gt_wp: torch.Tensor, aux_wp: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss as weighted combination of enabled loss components.
        
        Args:
            pred: Model predictions [pred_wp, weights, muy, sigma]
            gt_wp: Ground truth waypoints (B, T, 2)
            aux_wp: Auxiliary waypoints (B, N, T, 2)
        
        Returns:
            total_loss: Scalar loss tensor
            logs: Dict of individual loss values for logging
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        logs = {}
        
        for loss_name in self.enabled_losses:
            loss_module = self.loss_modules[loss_name]
            loss_val = loss_module(pred, gt_wp, aux_wp)
            if loss_val is None: continue
            
            # Apply coefficient
            coeff = self.loss_coeffs.get(loss_name, 1.0)
            weighted_loss = coeff * loss_val
            
            # Accumulate
            total_loss = total_loss + weighted_loss
            logs[loss_name] = loss_val.detach()
        
        return total_loss, logs

    
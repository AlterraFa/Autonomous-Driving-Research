import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from model.ImprovedVENL.impl.model import ImprovedVENL

class NavLoss(nn.Module):
    def __init__(
        self, 
        nav_model  : ImprovedVENL, 
        delta      : float, 
        target_sep : list, 
        target_std : list, 
        loss_coeffs: dict,
        device     : torch.device, 
        pad_value  : float,
    ):
        super().__init__()
        
        self.model = nav_model
        self.device = device
        self.target_sep    = torch.as_tensor(target_sep, dtype = torch.float32).to(device)
        self.target_std    = torch.as_tensor(target_std, dtype = torch.float32).to(device)
        self.delta         = delta
        self.components    = self.model.components
        self.pad_value     = pad_value
        
        self.mask_sep = torch.triu(
            torch.ones(*[self.components] * 2, dtype = torch.float).to(device),
            diagonal = 1
        )
        self.mask_sep    = self.mask_sep.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.gaussian_fn = self.model.gaussian_function

        
        default_coeffs = {
            "nll": 1.0, "mse": 1.0, "std_reg": 1.0, "entropy": 0.0, "repulsion": 0.0,
            "l1_gmm": 0.0, "l2_gmm": 0.0, # Regularization for GMM mixing weights
            "l1_model": 0.0, "l2_model": 0.0001, # Regularization for Model weights
            "dir": 0.0, "lon": 0.0, "lat": 0.0
        }

        self.loss_coeffs = {**default_coeffs, **loss_coeffs}
    
    def _compute_gmm_nll(self, pred: list[torch.Tensor], aux_wp: torch.Tensor):
        _, weights, muy, sigma = pred
        
        # -- Comnpute gaussian probability per mode
        gmm_prob_per_mode = self.gaussian_fn(sample = aux_wp, parameters = (weights, muy, sigma))

        # -- Masked out invalid prob
        mask_aux          = aux_wp.sum((-1, -2), keepdim = True) != (torch.prod(torch.tensor(aux_wp.shape[2:])) * self.pad_value)
        masked_gmm_prob   = gmm_prob_per_mode * mask_aux

        # -- Compute the total probability and nll loss
        mask_sum = mask_aux.sum(1)
        safe_denominator       = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum * 2)
        total_gmm_prob         = (masked_gmm_prob.sum(1) / safe_denominator).sum(1)
        total_gmm_prob_clamped = torch.clamp(total_gmm_prob, min = 1e-6)
        nll_loss               = (-torch.log(total_gmm_prob_clamped)).mean()
        
        return nll_loss
    
        
    def _compute_dist_loss(self, pred: list[torch.Tensor], gt_wp: torch.Tensor):
        pred_wp, *_ = pred
        
        mse_loss = F.huber_loss(pred_wp, gt_wp, delta = self.delta, reduction = "mean")
        return mse_loss

    def _compute_dir_loss(self, pred: list[torch.Tensor], gt_wp: torch.Tensor):
        pred_wp, *_ = pred
        
        pred_dir = pred_wp[:, 1:] - pred_wp[:, :-1]
        gt_dir   = gt_wp[:, 1:] - gt_wp[:, :-1]

        pred_dir = F.normalize(pred_dir, p = 2, dim = -1)
        gt_dir   = F.normalize(gt_dir, p = 2, dim = -1)

        cos_sim = (pred_dir * gt_dir).sum(dim = -1)
        return (1.0 - cos_sim).mean()
    def _compute_smoothness_loss(self, pred):
        pred_wp, *_ = pred
        
        v = pred_wp[:, 1:] - pred_wp[:, :-1]
        a = v[:, 1:] - v[:, :-1]
        j = a[:, 1:] - a[:, :-1]
    
        return a.pow(2).mean() + j.pow(2).mean()

    def _compute_seperate_dist_loss(self, pred: list[torch.Tensor], gt_wp: torch.Tensor):
        pred_wp, *_ = pred
    
        tangent = F.normalize(gt_wp[:, 1:] - gt_wp[:, :-1], p=2, dim=-1)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        
        error = pred_wp[:, 1:] - gt_wp[:, 1:]
        
        lat_err = (error * normal).sum(dim=-1).abs().mean()
        lon_err = (error * tangent).sum(dim=-1).abs().mean()

        return lat_err, lon_err
    
    def _compute_regularization(self, pred):
        _, weights, _, sigma = pred
        
        # -- Sigma Regularization (Target spread)
        t_std_view = torch.exp(self.target_std).view(1, 1, -1, 1).expand_as(sigma)
        std_reg    = F.mse_loss(sigma, t_std_view)
        # std_reg    = torch.clamp(std_reg, max = 3.0)

        # -- Weight Entropy (encourage confident component selection)
        probs = torch.exp(weights)
        entropy_w = -(probs * weights).sum(dim=1).mean()
        
        # -- GMM weight regularization
        l1_gmm = weights.abs().mean()
        l2_gmm = weights.pow(2.0).mean()

        return std_reg, entropy_w, l1_gmm, l2_gmm
    
    def _compute_model_reg(self):
        # Note: Usually handled by optimizer weight_decay, but keeping your custom logic
        weight_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "weight" in n]
        
        if not weight_params: return 0.0, 0.0
        
        l1_norm = sum(p.abs().mean() for p in weight_params)
        l2_norm = sum(p.pow(2.0).mean() for p in weight_params)
        return l1_norm, l2_norm 
    
    def _compute_gmm_assignment(self, pred, gt_wp, aux_wp):
        _, weights, muy, sigma = pred  # muy: (B, K, T, 2), weights: (B, K, T)
        B, K, T, _ = muy.shape
        device = muy.device

        mapped_targets = torch.zeros((B, K, T, 2), device=device)
        valid_mask = torch.zeros((B, K, T), device=device, dtype=torch.bool)

        center_threshold = 0.5
        for b in range(B):
            heading = gt_wp[b, -1] - gt_wp[b, 0]
            if torch.norm(heading) < 1e-4: heading = torch.tensor([0.0, 1.0], device=device)
            
            lefts, rights, centers = [], [], []
            
            for n in range(aux_wp.shape[1]):
                traj = aux_wp[b, n]
                
                if torch.all(traj[0] == self.pad_value):
                    continue

                offset = traj[0] - gt_wp[b, 0]
                dist   = torch.norm(offset)
                side   = heading[0] * offset[1] - heading[1] * offset[0]

                if dist < center_threshold:
                    centers.append((dist, traj))
                elif side > 0:
                    lefts.append((dist, traj))
                else:
                    rights.append((dist, traj))

            centers.sort(key = lambda x: x[0])
            lefts.sort(key = lambda x: x[0])
            rights.sort(key = lambda x: x[0])

            
            if centers:
                mapped_targets[b, 0] = centers[0][1]
                valid_mask[b, 0, :] = True

            left_slots = [i for i in range(1, K) if i % 2 != 0]
            for i, slot_idx in enumerate(left_slots):
                if i < len(lefts):
                    mapped_targets[b, slot_idx] = lefts[i][1]
                    valid_mask[b, slot_idx, :] = True

            right_slots = [i for i in range(2, K) if i % 2 == 0]
            for i, slot_idx in enumerate(right_slots):
                if i < len(rights):
                    mapped_targets[b, slot_idx] = rights[i][1]
                    valid_mask[b, slot_idx, :] = True

        log_probs = self._multivariate_log_prob(mapped_targets, muy, sigma) # (B, K, T)
        total_log_prob = weights + log_probs # (B, K, T)
        nll_all = -total_log_prob

        if valid_mask.any():
            matched_nll_loss = nll_all[valid_mask].mean()
        else:
            matched_nll_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return matched_nll_loss

    def _multivariate_log_prob(self, sample, muy, sigma):
        eps = 1e-6
        z_score = (sample - muy) / (sigma + eps)
        squared_diff = z_score ** 2
        
        squared_diff = torch.clamp(squared_diff, max=100.0) 
        exponent_term = -0.5 * squared_diff.sum(dim=-1) # Sum X and Y

        log_2pi = 1.837877
        log_sigma_sum = torch.log(sigma + eps).sum(dim=-1)
        log_norm_const = - (log_2pi + log_sigma_sum)

        return exponent_term + log_norm_const


    def forward(self, pred, gt_wp, aux_wp):
        
        # -- 1. Task Losses
        mse_loss = self._compute_dist_loss(pred, gt_wp)
        nll_loss = self._compute_gmm_assignment(pred, gt_wp, aux_wp)
        
        # -- More waypoint loss
        dir_loss = self._compute_dir_loss(pred, gt_wp)
        lat_loss, lon_loss = self._compute_seperate_dist_loss(pred, gt_wp)
        
        # -- 2. GMM Structure Regularization
        std_reg, ent_loss, l1_gmm, l2_gmm = self._compute_regularization(pred)
        
        # -- 3. Model Weights Regularization
        l1_model, l2_model = self._compute_model_reg()
        
        # -- 4. Combine them
        total_loss = (
            self.loss_coeffs["nll"] * nll_loss +
            self.loss_coeffs["mse"] * mse_loss +
            self.loss_coeffs["dir"] * dir_loss +
            self.loss_coeffs["lat"] * lat_loss +
            self.loss_coeffs["lon"] * lon_loss +
            
            # self.loss_coeffs["std_reg"] * std_reg +
            self.loss_coeffs["entropy"] * ent_loss +
            # self.loss_coeffs["l1_gmm"] * l1_gmm +
            # self.loss_coeffs["l2_gmm"] * l2_gmm +
            self.loss_coeffs["l1_model"] * l1_model +
            self.loss_coeffs["l2_model"] * l2_model
        )
                      
        return total_loss, {
            "Huber": mse_loss.detach(), "NLL": nll_loss.detach(), "Entropy": ent_loss.detach(),
            "Lat": lat_loss.detach(), "Lon": lon_loss.detach(), "Dir": dir_loss.detach()
        }
    
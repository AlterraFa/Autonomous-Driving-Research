import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from model.JEPA_VENL.impl.jepa_nav import UnifiedJEPANav

class NavLoss(nn.Module):
    def __init__(
        self, 
        nav_model  : UnifiedJEPANav, 
        delta      : float, 
        target_sep : list, 
        target_std : list, 
        loss_coeffs: dict,
        device     : torch.device, 
        pad_value  : float,
        full_finetune = False
    ):
        super().__init__()
        
        self.model = nav_model
        self.device = device
        self.target_sep    = torch.as_tensor(target_sep, dtype = torch.float32).to(device)
        self.target_std    = torch.as_tensor(target_std, dtype = torch.float32).to(device)
        self.delta         = delta
        self.full_finetune = full_finetune
        self.components    = self.model.readout.components
        self.pad_value     = pad_value
        
        self.mask_sep = torch.triu(
            torch.ones(*[self.components] * 2, dtype = torch.float).to(device),
            diagonal = 1
        )
        self.mask_sep    = self.mask_sep.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.gaussian_fn = self.model.readout.gaussian_function

        
        default_coeffs = {
            "nll": 1.0, "mse": 1.0, "std_reg": 1.0, "entropy": 0.01, "repulsion": 0.1,
            "l1_gmm": 0.0, "l2_gmm": 0.0, # Regularization for GMM mixing weights
            "l1_model": 0.0, "l2_model": 0.0001 # Regularization for Model weights
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
    
    def _compute_soft_wta_nll(self, pred: list[torch.Tensor], aux_wp: torch.Tensor, temperature: float = 2.0):
        """
        Computes NLL using Soft Winner-Take-All.
        Instead of optimizing only the single best head, we weight the loss of all heads
        based on their distance to the GT.
        """
        _, weights, muy, sigma = pred
        # muy: (B, K, T, 2)
        # sigma: (B, K, T, 2)
        # weights: (B, K, 1)
        # aux_wp: (B, M, T, 2)
        
        B, K, T, _ = muy.shape
        _, M, _, _ = aux_wp.shape 

        # 1. Identify Valid Data (True = Real, False = Padding)
        valid_mask = aux_wp[..., 0] != self.pad_value # (B, M, T)
        path_has_data = valid_mask.any(dim=-1)  # (B, M)

        # 2. Calculate Distance Matrix (Squared L2) between ALL heads and ALL GT paths
        # Expand preds to match M paths
        mu_exp = muy.unsqueeze(2).expand(-1, -1, M, -1, -1) # (B, K, M, T, 2)
        # Expand GT to match K heads
        gt_exp = aux_wp.unsqueeze(1).expand(-1, K, -1, -1, -1) # (B, K, M, T, 2)
        
        # Mask the distance calculation
        dist_mask = valid_mask.view(B, 1, M, T).float()
        diff_sq = (mu_exp - gt_exp).pow(2).sum(dim=-1)
        masked_diff_sq = diff_sq * dist_mask
        
        # Sum over time to get total trajectory distance
        dist_matrix = masked_diff_sq.sum(dim=-1) # (B, K, M)

        # 3. Loop over GT Paths (M)
        total_nll = 0.0
        total_matches = 0

        # Pre-clamp sigma to prevent singularity/collapse (Strategy #1 included here)
        sigma = torch.clamp(sigma, min=0.05) 

        for m in range(M):
            valid_indices = path_has_data[:, m] # (B,) boolean
            if not valid_indices.any():
                continue

            # -- Step A: Calculate Soft Assignments --
            # Get distances for this specific GT path index m
            dists = dist_matrix[:, :, m] # (B, K)
            
            # For numerical stability in softmax, subtract min
            dists_stable = dists - dists.min(dim=1, keepdim=True)[0]
            
            # Convert distances to weights: Closer heads get higher weight
            # High temperature = smoother distribution, Low temp = harder WTA
            assignment_weights = F.softmax(-dists_stable / temperature, dim=1) # (B, K)
            
            # Stop gradients flowing into the assignment weights themselves? 
            # Usually we want the heads to move closer, so we keep gradients.
            # If training becomes unstable, uncomment: assignment_weights = assignment_weights.detach()

            # -- Step B: Compute NLL for ALL K heads against this GT path --
            # Get current GT path
            gt_path = aux_wp[:, m, :, :] # (B, T, 2)
            timestep_mask = valid_mask[:, m, :].float() # (B, T)

            # Expand for broadcasting against K heads
            gt_path_k = gt_path.unsqueeze(1).expand(-1, K, -1, -1)     # (B, K, T, 2)
            mask_k    = timestep_mask.unsqueeze(1).expand(-1, K, -1)   # (B, K, T)

            # TRICK: Replace Padding with Prediction for all heads to avoid NaNs
            safe_gt = gt_path_k.clone().to(torch.bfloat16)
            safe_gt[mask_k == 0] = muy[mask_k == 0]

            # -- Gaussian NLL Calculation (Vectorized over K) --
            # 1. Norm Const
            norm_const = (1.0 / (torch.sqrt(torch.tensor(2.0 * torch.pi, device=self.device)) * sigma)).prod(dim=-1)
            
            # 2. Exponent
            mahalanobis_sq = ((safe_gt - muy) / sigma) ** 2
            exp_term = torch.exp(-0.5 * mahalanobis_sq.sum(dim=-1))
            
            # 3. Prob Density
            probs = norm_const * exp_term
            probs_clamped = torch.clamp(probs, min=1e-8)
            nll_trajectory = -torch.log(probs_clamped) # (B, K, T)

            # Sum NLL over time steps (apply mask)
            nll_trajectory_sum = (nll_trajectory * mask_k).sum(dim=2) # (B, K)

            # Add the NLL of the mixing weight (network confidence)
            # The network predicted probability for head k is weights[:, k]
            nll_mixing = -torch.log(weights.squeeze(-1) + 1e-6) # (B, K)
            
            total_head_loss = nll_trajectory_sum + nll_mixing # (B, K)

            # -- Step C: Weighted Average --
            # We weight the loss of each head by how likely that head was the "correct" one
            weighted_loss = (total_head_loss * assignment_weights).sum(dim=1) # (B,)

            # Accumulate
            mask_batch = valid_indices.float()
            total_nll += (weighted_loss * mask_batch).sum()
            total_matches += mask_batch.sum()

        avg_loss = total_nll / (total_matches + 1e-6)
        return avg_loss
    
    def _compute_nll(self, pred: list[torch.Tensor], aux_wp: torch.Tensor):
        _, weights, muy, sigma = pred
        
        # -- (B, num_aux)
        B, C, _, _ = muy.shape
        _, N, _, _ = aux_wp.shape
        expand_aux = aux_wp.unsqueeze(2).expand(-1, -1, C, -1, -1)
        expand_muy = muy.unsqueeze(1).expand(-1, N, -1, -1, -1)
        
        # -- 3 aux correspond to 6 wp possible components
        dist_to_aux = torch.norm(expand_aux - expand_muy, dim = (-1, -2), p = 2)
        mask_aux    = aux_wp.sum((-1, -2), keepdim = True) != (torch.prod(torch.tensor(aux_wp.shape[2:])) * self.pad_value)

        dist_to_aux = dist_to_aux.detach().cpu().numpy()
        mask_aux    = mask_aux.detach().cpu().numpy()

        total_nll     = 0.0
        valid_batches = 0

        for b in range(B):
            valid_aux_indices = np.where(mask_aux[b])[0]
            
            if len(valid_aux_indices) == 0:
                continue
            
            cost_matrix = dist_to_aux[b, valid_aux_indices, :]
            
            row_idx, col_idx = linear_sum_assignment(cost_matrix)

            actual_aux_indices = valid_aux_indices[row_idx]
            matched_mode_indices = col_idx
            
            target = aux_wp[b, actual_aux_indices] 
            mu  = muy[b, matched_mode_indices]
            sig = sigma[b, matched_mode_indices]
            w   = weights[b, matched_mode_indices]

            var = torch.clamp(sig ** 2, min = 1e-5)
            gauss_nll = 0.5 * (torch.log(var) + (target - mu)**2 / var)
            gauss_nll = gauss_nll.sum((-1, -2))
            
            weight_nll = -torch.log(w + 1e-6) 

            batch_loss = gauss_nll + weight_nll
            
            total_nll     += batch_loss.sum()
            valid_batches += 1
        
        return total_nll / valid_batches
        
    def _compute_dist_loss(self, pred: list[torch.Tensor], gt_wp: torch.Tensor):
        pred_wp, *_ = pred
        
        mse_loss = F.huber_loss(pred_wp, gt_wp, delta = self.delta, reduction = "mean")
        return mse_loss
    
    def _compute_regularization(self, pred):
        _, weights, muy, sigma = pred
        
        # -- Repulsion (Separation of means)
        a = muy.unsqueeze(1) # Shape: (B, 1, K, T, 2)
        b = muy.unsqueeze(2) # Shape: (B, K, 1, T, 2)
        dist_sq = (a - b) ** 2
        
        # -- Broadcast target_sep: (1, 1, 1, T, 1)
        ts_view = self.target_sep.view(1, 1, 1, -1, 1).expand_as(dist_sq)
        sep_coeff = torch.exp(-dist_sq / (2 * ts_view))
        
        # -- Apply mask (only upper triangle to avoid double counting or self-compare)
        sep_coeff = sep_coeff * self.mask_sep.expand_as(sep_coeff)
        repulsion_loss = sep_coeff.sum((1, 2, 3, 4)).mean() / self.components

        # -- Sigma Regularization (Target spread)
        t_std_view = torch.exp(self.target_std).view(1, 1, -1, 1).expand_as(sigma)
        std_reg    = F.mse_loss(sigma, t_std_view)
        # std_reg    = torch.clamp(std_reg, max = 3.0)

        # -- Weight Entropy (encourage confident component selection)
        weights_clamped = torch.clamp(weights, min = 1e-6)
        entropy_w = -(weights.squeeze() * torch.log(weights_clamped.squeeze())).sum(dim=1).mean()
        
        # -- GMM weight regularization
        l1_gmm = weights.abs().mean()
        l2_gmm = weights.pow(2.0).mean()

        return repulsion_loss, std_reg, entropy_w, l1_gmm, l2_gmm
    
    def _compute_model_reg(self):
        # Note: Usually handled by optimizer weight_decay, but keeping your custom logic
        if self.full_finetune:
            weight_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "weight" in n]
        else:
            weight_params = [p for n, p in self.model.readout.named_parameters() if p.requires_grad and "weight" in n]
        
        if not weight_params: return 0.0, 0.0
        
        l1_norm = sum(p.abs().mean() for p in weight_params)
        l2_norm = sum(p.pow(2.0).mean() for p in weight_params)
        return l1_norm, l2_norm 

    def forward(self, pred, gt_wp, aux_wp):
        
        # -- 1. Task Losses
        mse_loss = self._compute_dist_loss(pred, gt_wp)
        nll_loss = self._compute_gmm_nll(pred, aux_wp)
        
        
        # -- 2. GMM Structure Regularization
        rep_loss, std_reg, ent_loss, l1_gmm, l2_gmm = self._compute_regularization(pred)
        
        # -- 3. Model Weights Regularization
        l1_model, l2_model = self._compute_model_reg()
        
        # -- 4. Combine them
        total_loss = (
            self.loss_coeffs["nll"] * nll_loss +
            self.loss_coeffs["mse"] * mse_loss +
            self.loss_coeffs["std_reg"] * std_reg +
            self.loss_coeffs["entropy"] * ent_loss +
            self.loss_coeffs["repulsion"] * rep_loss +
            self.loss_coeffs["l1_gmm"] * l1_gmm +
            self.loss_coeffs["l2_gmm"] * l2_gmm +
            self.loss_coeffs["l1_model"] * l1_model +
            self.loss_coeffs["l2_model"] * l2_model
        )
                      
        return total_loss, {
            "Huber": mse_loss.detach(), "NLL": nll_loss.detach(), "Repulsion": rep_loss.detach(), 
            "Std_Reg": std_reg.detach(), "Entropy": ent_loss.detach()
        }
    
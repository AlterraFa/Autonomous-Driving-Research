
import os, sys
import torch
import yaml
import torch.nn.functional as F
import numpy as np

from tqdm.auto import tqdm
from model.SingleVENL.model import SingleVENL

from torch import optim
from torch.utils.data import DataLoader

FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../.."))


with open(FILE_DIR + "/model_cfg.yaml", "r") as f:
    config = yaml.safe_load(f)

venl_config = config["config"]
venl_loss = config["loss_contrib"]

mse_contrib       = config['loss_contrib']['mse_contrib']
nll_contrib       = config['loss_contrib']['nll_contrib']
std_reg_contrib   = config['loss_contrib']['std_reg_contrib']
entropy_contrib   = config['loss_contrib']['entropy_contrib']
repulsion_contrib = config['loss_contrib']['repulsion_contrib']
l1_contrib        = config['loss_contrib']['l1_contrib']
l2_contrib        = config['loss_contrib']['l2_contrib']
l1_weight_contrib = config['loss_contrib']['l1_weight_contrib']
l2_weight_contrib = config['loss_contrib']['l2_weight_contrib']

l1         = float(config['model']['l1'])
l2         = float(config['model']['l2'])
l1_weight  = float(config['model']['l1_weight'])
l2_weight  = float(config['model']['l2_weight'])
target_std = torch.tensor(config['model']['target_std'])   # List
target_sep = torch.tensor(config['model']['target_sep'])   # List

# Optional waypoint normalization stats (per-waypoint, per-dim)
wp_mean_cfg = config['model'].get('wp_mean', None)
wp_std_cfg  = config['model'].get('wp_std',  None)


def single_epoch_training(model: SingleVENL, loader: DataLoader, lr_scheduler, wd_scheduler, optimizer: optim, log_stats):
    model.train()
    device = next(model.parameters()).device


    for images, controls in log_stats.batch_iterator(loader):
        optimizer.zero_grad(set_to_none=True)

        images = {name: image.to(device) for name, image in images.items()}
        gt     = controls['midlane_wp'].to(device) if model.output_names[0] == "waypoint" else controls['steer'].unsqueeze(1).to(device)
        aux_gt = controls['aux_wp'].to(device)

        determ, weights, muy, sigma = model(**images)

        # === NLL in normalized space (if stats provided) ===
        use_norm = (wp_mean_cfg is not None) and (wp_std_cfg is not None)
        if use_norm:
            wp_mean = torch.tensor(wp_mean_cfg, dtype=gt.dtype, device=device)  # (N,2)
            wp_std  = torch.tensor(wp_std_cfg,  dtype=gt.dtype, device=device)  # (N,2)

            # reshape for broadcasting: (1,1,N,2)
            mean_b = wp_mean.view(1, 1, model.num_waypoints, 2)
            std_b  = wp_std.view(1, 1, model.num_waypoints, 2)

            # normalize aux ground truth
            aux_in = (aux_gt - mean_b) / (std_b + 1e-6)
            
            muy_in   = muy
            sigma_in = sigma
        else:
            aux_in, muy_in, sigma_in = aux_gt, muy, sigma

        gmm_prob_per_mode = model.gaussian_function(aux_in, (weights, muy_in, sigma_in))
        mask_aux          = aux_gt.abs().sum((-1, -2), keepdim = True) != 0
        masked_gmm_prob   = gmm_prob_per_mode * mask_aux
        total_gmm_prob    = (masked_gmm_prob.sum(1) / mask_aux.sum(1)).sum(1)


        # Loss Components
        nll_loss = (-torch.log(total_gmm_prob + 1e-20)).mean()

        if use_norm and model.output_names[0] == 'waypoint':
            mean_wp = wp_mean.view(1, model.num_waypoints, 2)
            std_wp  = wp_std.view(1, model.num_waypoints, 2)
            gt_m     = (gt - mean_wp) / (std_wp + 1e-6)
            mse_loss = F.mse_loss(determ, gt_m)
        else:
            mse_loss = F.mse_loss(determ, gt)
            

        #  Extra loss to encourage correct GMM Behavior
        a = muy.unsqueeze(1)
        b = muy.unsqueeze(2)
        dist_sq   = (a - b) ** 2 # A square matrix distance
        sep_coeff = torch.exp(- dist_sq / (2 * target_sep.view(1, 1, 1, -1, 1).expand_as(dist_sq).to(device))) # Each waypoint will have its own target seperation, hence (1, 1, 1, -1, 1)
        if "mask_sep" not in locals():
            mask_sep  = torch.triu(torch.ones(model.components, model.components, dtype = torch.float).to(device), diagonal = 1)
            mask_sep  = mask_sep.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        sep_coeff = sep_coeff * mask_sep.expand_as(sep_coeff)
        repulsion_loss = sep_coeff.sum((1, 2, 3, 4)).mean() / model.components # Sum over all gaussians and normalize by the number of components
        std_reg = F.mse_loss(torch.log(sigma), target_std.view(1, 1, -1, 1).expand_as(sigma).to(device))
        entropy_w = -(weights.squeeze() * torch.log(weights.squeeze() + 1e-9)).sum(dim=1).mean()
        l1_weights_norm = weights.abs().mean()
        l2_weights_norm = weights.pow(2.0).mean()

        # Model weight regularization
        weightParams = [p for n, p in model.named_parameters() if p.requires_grad and "weight" in n]
        l1Norm = sum(p.abs().mean() for p in weightParams)
        l2Norm = sum(p.pow(2.0).mean() for p in weightParams)

        # Total weighted loss
        loss = (
            nll_contrib       * nll_loss +
            mse_contrib       * mse_loss +
            std_reg_contrib   * std_reg +
            entropy_contrib   * entropy_w +
            repulsion_contrib * repulsion_loss +
            l1_weight_contrib * l1_weights_norm * l1_weight +
            l2_weight_contrib * l2_weights_norm * l2_weight +
            l1_contrib        * l1Norm * l1 +
            l2_contrib        * l2Norm * l2
        )

        loss.backward()

        optimizer.step()
        lr_scheduler.step() 
        wd_scheduler.step()

        # Debug: monitor gradients to deterministic head
        det_head_grads = [p.grad.abs().mean().item() for p in model.determ_head.parameters() if p.grad is not None]
        fus_grads = [p.grad.abs().mean().item() for p in model.fusion_projector.parameters() if p.grad is not None]
        det_head_grad_mean = float(np.mean(det_head_grads)) if det_head_grads else 0.0
        fus_grad_mean = float(np.mean(fus_grads)) if fus_grads else 0.0

        log_stats.log_batch({
            "Total": loss.item(),
            "MSE": mse_loss.item(),
            "NLL": nll_loss.item(),
            "STD": std_reg.item(),
            "ENT": entropy_w.item(),
            "DH_Grad": det_head_grad_mean,
            "Fus_Grad": fus_grad_mean,
        })


    del images, gt, determ, weights, muy, sigma
    torch.cuda.empty_cache()


def single_epoch_val(model: SingleVENL, loader: DataLoader, log_stats):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for images, controls in log_stats.batch_iterator(loader):

            images = {name: image.to(device) for name, image in images.items()}
            gt     = controls['midlane_wp'].to(device) if model.output_names[0] == "waypoint" else controls['steer'].unsqueeze(1).to(device)
            aux_gt = controls['aux_wp'].to(device)


            determ, weights, muy, sigma = model(**images)

            # === NLL in normalized space (if stats provided) ===
            use_norm = (wp_mean_cfg is not None) and (wp_std_cfg is not None)
            if use_norm:
                wp_mean = torch.tensor(wp_mean_cfg, dtype=gt.dtype, device=device)  # (N,2)
                wp_std  = torch.tensor(wp_std_cfg,  dtype=gt.dtype, device=device)  # (N,2)

                # reshape for broadcasting: (1,1,N,2)
                mean_b = wp_mean.view(1, 1, model.num_waypoints, 2)
                std_b  = wp_std.view(1, 1, model.num_waypoints, 2)

                aux_in = (aux_gt - mean_b) / (std_b + 1e-6)
                muy_in   = muy
                sigma_in = sigma
            else:
                aux_in, muy_in, sigma_in = aux_gt, muy, sigma

            gmm_prob_per_mode = model.gaussian_function(aux_in, (weights, muy_in, sigma_in))
            mask_aux          = aux_gt.abs().sum((-1, -2), keepdim = True) != 0
            masked_gmm_prob   = gmm_prob_per_mode * mask_aux
            total_gmm_prob    = (masked_gmm_prob.sum(1) / mask_aux.sum(1)).sum(1)

            # Loss Components
            nll_loss = (-torch.log(total_gmm_prob + 1e-20)).mean()

            if use_norm and model.output_names[0] == 'waypoint':
                mean_wp = wp_mean.view(1, model.num_waypoints, 2)
                std_wp  = wp_std.view(1, model.num_waypoints, 2)
                gt_m     = (gt - mean_wp) / (std_wp + 1e-6)
                mse_loss = F.mse_loss(determ, gt_m)
            else:
                mse_loss = F.mse_loss(determ, gt)

            #  Extra loss to encourage correct GMM Behavior
            a = muy.unsqueeze(1)
            b = muy.unsqueeze(2)
            dist_sq   = (a - b) ** 2 # A square matrix distance
            sep_coeff = torch.exp(- dist_sq / (2 * target_sep.view(1, 1, 1, -1, 1).expand_as(dist_sq).to(device))) # Each waypoint will have its own target seperation, hence (1, 1, 1, -1, 1)
            if "mask_sep" not in locals():
                mask_sep  = torch.triu(torch.ones(model.components, model.components, dtype = torch.float).to(device), diagonal = 1)
                mask_sep  = mask_sep.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            sep_coeff = sep_coeff * mask_sep.expand_as(sep_coeff)
            repulsion_loss = sep_coeff.sum((1, 2, 3, 4)).mean() / model.components # Sum over all gaussians and normalize by the number of components
            std_reg = F.mse_loss(torch.log(sigma), target_std.view(1, 1, -1, 1).expand_as(sigma).to(device))
            entropy_w = -(weights.squeeze() * torch.log(weights.squeeze() + 1e-9)).sum(dim=1).mean()
            l1_weights_norm = weights.abs().mean()
            l2_weights_norm = weights.pow(2.0).mean()

            # Model weight regularization
            weightParams = [p for n, p in model.named_parameters() if p.requires_grad and "weight" in n]
            l1Norm = sum(p.abs().mean() for p in weightParams)
            l2Norm = sum(p.pow(2.0).mean() for p in weightParams)

            # Total weighted loss
            loss = (
                nll_contrib       * nll_loss +
                mse_contrib       * mse_loss +
                std_reg_contrib   * std_reg +
                entropy_contrib   * entropy_w +
                repulsion_contrib * repulsion_loss +
                l1_weight_contrib * l1_weights_norm * l1_weight +
                l2_weight_contrib * l2_weights_norm * l2_weight +
                l1_contrib        * l1Norm * l1 +
                l2_contrib        * l2Norm * l2
            )

            # Debug: monitor gradients to deterministic head
            det_head_grads = [p.grad.abs().mean().item() for p in model.determ_head.parameters() if p.grad is not None]
            fus_grads = [p.grad.abs().mean().item() for p in model.fusion_projector.parameters() if p.grad is not None]
            det_head_grad_mean = float(np.mean(det_head_grads)) if det_head_grads else 0.0
            fus_grad_mean = float(np.mean(fus_grads)) if fus_grads else 0.0

            log_stats.log_batch({
                "Total": loss.item(),
                "MSE": mse_loss.item(),
                "NLL": nll_loss.item(),
                "STD": std_reg.item(),
                "ENT": entropy_w.item(),
                "DH_Grad": det_head_grad_mean,
                "Fus_Grad": fus_grad_mean,
            }, phase = "val")


    del images, gt, determ, weights, muy, sigma
    torch.cuda.empty_cache()
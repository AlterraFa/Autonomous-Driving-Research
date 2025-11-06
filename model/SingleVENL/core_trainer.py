
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

mse_contrib       = float(venl_loss.get("mse_contrib", 0.0))
nll_contrib       = float(venl_loss.get("nll_contrib", 0.0))
std_reg_contrib   = float(venl_loss.get("std_reg_contrib", 0.0))
entropy_contrib   = float(venl_loss.get("entropy_contrib", 0.0))
repulsion_contrib = float(venl_loss.get("repulsion_contrib", 0.0))
l1_weight_contrib = float(venl_loss.get("l1_weight_contrib", 0.0))
l2_weight_contrib = float(venl_loss.get("l2_weight_contrib", 0.0))
l1_contrib        = float(venl_loss.get("l1_contrib", 0.0))
l2_contrib        = float(venl_loss.get("l2_contrib", 0.0))

l1         = float(venl_config["l1"])
l2         = float(venl_config["l2"])
l1_weight  = float(venl_config["l1_weight"])
l2_weight  = float(venl_config["l2_weight"])
target_std = torch.tensor(venl_config["target_std"])
target_sep = torch.tensor(venl_config["target_sep"])


def single_epoch_training(model: SingleVENL, loader: DataLoader, optimizer: optim):
    model.train()
    device = next(model.parameters()).device

    trainBar = tqdm(loader, desc="Train", position=1, leave=False)

    # Metrics now include loss + gradient stats
    trainMetrics = {
        "Total": 0,
        "MSE": 0,
        "NLL": 0,
        "STD Reg": 0,
        "Weights Entropy": 0,
        "Repulsion": 0,
        "Grad_Routed": 0,
        "Grad_Unrouted": 0,
        "Grad_Cam": 0,
    }

    for batch_idx, (images, controls) in enumerate(trainBar):
        optimizer.zero_grad(set_to_none=True)

        images = {name: image.to(device) for name, image in images.items()}
        gt     = controls['midlane_wp'].to(device) if model.output_names[0] == "waypoint" else controls['steer'].unsqueeze(1).to(device)
        aux_gt = controls['aux_wp'].to(device)

        determ, weights, muy, sigma = model(**images)
        gmm_prob_per_mode = model.gaussian_function(aux_gt, (weights, muy, sigma))
        mask_aux          = aux_gt.abs().sum((-1, -2), keepdim = True) != 0
        masked_gmm_prob   = gmm_prob_per_mode * mask_aux
        total_gmm_prob    = (masked_gmm_prob.sum(1) / mask_aux.sum(1)).sum(1)


        # Loss Components
        nll_loss = (-torch.log(total_gmm_prob + 1e-20)).mean()
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

        # === Gradient Monitoring ===
        def grad_mean_abs(module):
            grads = [p.grad.abs().mean().item() for p in module.parameters() if p.grad is not None]
            return float(np.mean(grads)) if grads else 0.0

        grad_cam0 = grad_mean_abs(model.cam_backbone)
        grad_unrouted = grad_mean_abs(model.unrouted_backbone)
        grad_routed = grad_mean_abs(model.routed_backbone)

        # Add gradient stats to metrics
        trainMetrics["Grad_Routed"]     += grad_routed
        trainMetrics["Grad_Unrouted"]   += grad_unrouted
        trainMetrics["Grad_Cam"]  += grad_cam0

        optimizer.step()

        # Update losses
        trainMetrics["Total"]           += loss.item()
        trainMetrics["MSE"]             += mse_loss.item()
        trainMetrics["NLL"]             += nll_loss.item()
        trainMetrics["STD Reg"]         += std_reg.item()
        trainMetrics["Repulsion"]       += repulsion_loss.item()
        trainMetrics["Weights Entropy"] += entropy_w.item()

        step = trainBar.n + 1
        avg_total = trainMetrics["Total"] / step
        avg_mse   = trainMetrics["MSE"] / step
        avg_nll   = trainMetrics["NLL"] / step
        avg_std   = trainMetrics["STD Reg"] / step
        avg_rep   = trainMetrics["Repulsion"] / step
        avg_ent   = trainMetrics["Weights Entropy"] / step

        # Gradient averages for postfix display
        avg_grad_routed   = trainMetrics["Grad_Routed"] / step
        avg_grad_unrouted = trainMetrics["Grad_Unrouted"] / step
        avg_grad_cam      = np.mean([trainMetrics["Grad_Cam"]]) / step

        trainBar.set_postfix({
            "Total": f"{avg_total:.3f}",
            "MSE": f"{avg_mse:.3f}",
            "NLL": f"{avg_nll:.3f}",
            "STD": f"{avg_std:.3f}",
            "ENT": f"{avg_ent:.3f}",
            "REP": f"{avg_rep:.3f}",
            "GradCam": f"{avg_grad_cam:.2e}",
            "GradUnR": f"{avg_grad_unrouted:.2e}",
            "GradR": f"{avg_grad_routed:.2e}"
        })

    # Normalize final metrics
    for key in trainMetrics.keys():
        trainMetrics[key] /= len(loader)

    del images, gt, determ, weights, muy, sigma
    torch.cuda.empty_cache()

    return trainMetrics


def single_epoch_val(model: SingleVENL, loader: DataLoader):
    model.eval()
    device = next(model.parameters()).device

    valBar = tqdm(loader, desc = "Val", position = 2, leave = False)
    valMetrics = {"Total": 0, "MSE": 0, "NLL": 0, "STD Reg": 0, "Weights Entropy": 0, "Repulsion": 0}
    with torch.no_grad():
        for images, controls in valBar:

            images = {name: image.to(device) for name, image in images.items()}
            gt     = controls['midlane_wp'].to(device) if model.output_names[0] == "waypoint" else controls['steer'].unsqueeze(1).to(device)
            aux_gt = controls['aux_wp'].to(device)


            determ, weights, muy, sigma = model(**images)
            gmm_prob_per_mode = model.gaussian_function(aux_gt, (weights, muy, sigma))
            mask_aux          = aux_gt.abs().sum((-1, -2), keepdim = True) != 0
            masked_gmm_prob   = gmm_prob_per_mode * mask_aux
            total_gmm_prob    = (masked_gmm_prob.sum(1) / mask_aux.sum(1)).sum(1)

            # Loss Components
            nll_loss = (-torch.log(total_gmm_prob + 1e-20)).mean()
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
            repulsion_loss = sep_coeff.sum((1, 2, 3, 4)).mean() # Sum over all gaussians
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



            valMetrics["Total"]           += loss.item()
            valMetrics["MSE"]             += mse_loss.item()
            valMetrics["NLL"]             += nll_loss.item()
            valMetrics["STD Reg"]         += std_reg.item()
            valMetrics["Repulsion"]       += repulsion_loss.item()
            valMetrics["Weights Entropy"] += entropy_w.item()

            avg_total = valMetrics["Total"] / (valBar.n + 1)
            avg_mse   = valMetrics["MSE"] / (valBar.n + 1)
            avg_nll   = valMetrics["NLL"] / (valBar.n + 1)
            avg_std   = valMetrics["STD Reg"] / (valBar.n + 1)
            avg_rep   = valMetrics["Repulsion"] / (valBar.n + 1)
            avg_ent   = valMetrics["Weights Entropy"] / (valBar.n + 1)

            valBar.set_postfix({
                "Total": f"{avg_total:.3f}",
                "MSE": f"{avg_mse:.3f}",
                "NLL": f"{avg_nll:.3f}",
                "STD": f"{avg_std:.3f}",
                "ENT": f"{avg_ent:.3f}",
                "REP": f"{avg_ent:.3f}"
            })

    valMetrics["Total"]           /= len(loader)
    valMetrics["MSE"]             /= len(loader)
    valMetrics["NLL"]             /= len(loader)
    valMetrics["STD Reg"]         /= len(loader)
    valMetrics["Repulsion"]       /= len(loader)
    valMetrics["Weights Entropy"] /= len(loader)

    del images, gt, determ, weights, muy, sigma
    torch.cuda.empty_cache()

    return valMetrics
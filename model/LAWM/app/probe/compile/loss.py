import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger

logger = Logger(__name__)


class FDATLoss(nn.Module):
    """Frenet-Decomposed Anisotropic Trajectory Loss.

    Decomposes waypoint prediction error into along-track (tangential) and
    cross-track (normal) components in the Frenet frame of the GT trajectory,
    then applies anisotropic penalties conditioned on scene context (gate).

    Inputs are expected in a space with isotropic axes — here, normalized pixel
    coordinates (u, v) ∈ [-1, 1] from camera projection. Using metric (x, y)
    with non-uniform normalization would distort the Frenet tangent.

    Key properties:
        - Cross-track error penalized much more than along-track (lane adherence)
        - Gate-conditioned dual mode: lane-following vs. intersection
        - Heading consistency via cosine similarity of direction vectors
        - Bathtub positional weighting: start + end waypoints weighted more
        - Endpoint anchor loss for intersection mode
        - Built-in smoothness regularizer (jerk penalty)
    """

    def __init__(
        self,
        alpha_lane: float = 20.0,
        beta_lane: float = 1.0,
        alpha_inter: float = 10.0,
        beta_inter: float = 3.0,
        lambda_heading: float = 2.0,
        lambda_endpoint: float = 5.0,
        lambda_smooth: float = 0.05,
        tau_start: float = 2.0,
        tau_end: float = 4.0,
        sl1_beta: float = 0.02,
    ):
        super().__init__()
        self.alpha_lane = alpha_lane
        self.beta_lane = beta_lane
        self.alpha_inter = alpha_inter
        self.beta_inter = beta_inter
        self.lambda_heading = lambda_heading
        self.lambda_endpoint = lambda_endpoint
        self.lambda_smooth = lambda_smooth
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.sl1_beta = sl1_beta

    def _frenet_decompose(self, pred, gt):
        """Project error vectors into the Frenet frame of the GT curve.

        Returns:
            e_s: along-track error  [B, T]
            e_d: cross-track error  [B, T]
        """
        T_vec = torch.zeros_like(gt)
        T_vec[:, 1:-1] = gt[:, 2:] - gt[:, :-2]
        T_vec[:, 0] = gt[:, 1] - gt[:, 0]
        T_vec[:, -1] = gt[:, -1] - gt[:, -2]
        T_vec = T_vec / (T_vec.norm(dim=-1, keepdim=True) + 1e-6)

        N_vec = torch.stack([-T_vec[..., 1], T_vec[..., 0]], dim=-1)

        e = pred - gt
        e_s = (e * T_vec).sum(dim=-1)
        e_d = (e * N_vec).sum(dim=-1)
        return e_s, e_d

    def _positional_weights(self, T, device):
        return self.get_bathtub_weights(T, self.tau_start, self.tau_end, device)

    @staticmethod
    def get_bathtub_weights(T: int, tau_start: float, tau_end: float, device) -> torch.Tensor:
        """Bathtub weight curve over T waypoints.

        Weight(i) = 1 + exp(-i/tau_start) + exp(-(T-1-i)/tau_end)

        High at both ends (start = immediate action, end = goal), low in the middle.
        """
        i = torch.arange(T, device=device, dtype=torch.float32)
        w = 1.0 + torch.exp(-i / max(tau_start, 1e-6)) + torch.exp(-(T - 1 - i) / max(tau_end, 1e-6))
        return w

    def _heading_loss(self, pred, gt):
        """Cosine-based heading error. Returns per-sample scalar [B]."""
        d_pred = pred[:, 1:] - pred[:, :-1]
        d_gt = gt[:, 1:] - gt[:, :-1]
        cos_sim = F.cosine_similarity(d_pred, d_gt, dim=-1)
        return (1.0 - cos_sim).mean(dim=-1)

    @staticmethod
    def _smoothness_loss(pred):
        """Second-order finite-difference penalty (jerk)."""
        diff = pred[:, 1:] - pred[:, :-1]
        return (diff[:, 1:] - diff[:, :-1]).pow(2).mean(dim=(1, 2))

    def forward(self, pred_wp, gt_wp, gate_score=None):
        """Compute FDAT loss.

        Args:
            pred_wp: predicted waypoints  [B, T, 2]
            gt_wp: ground-truth waypoints [B, T, 2]
            gate_score: CommandGate output [B, 1, 1] or [B] (optional)

        Returns:
            Dict of per-sample components (each [B]):
                frenet, heading, smooth, total
        """
        B, T, _ = pred_wp.shape

        e_s, e_d = self._frenet_decompose(pred_wp, gt_wp)
        w = self._positional_weights(T, pred_wp.device)

        sl1_d = F.smooth_l1_loss(
            e_d, torch.zeros_like(e_d), reduction="none", beta=self.sl1_beta,
        )
        sl1_s = F.smooth_l1_loss(
            e_s, torch.zeros_like(e_s), reduction="none", beta=self.sl1_beta,
        )

        l_lane = ((self.alpha_lane * sl1_d + self.beta_lane * sl1_s) * w).mean(dim=-1)

        l_inter = ((self.alpha_inter * sl1_d + self.beta_inter * sl1_s) * w).mean(dim=-1)
        l_endpoint = (pred_wp[:, -1] - gt_wp[:, -1]).pow(2).sum(dim=-1)
        l_inter = l_inter + self.lambda_endpoint * l_endpoint

        if gate_score is not None:
            g = gate_score.detach().view(B)
        else:
            g = torch.zeros(B, device=pred_wp.device)

        l_frenet = (1.0 - g) * l_lane + g * l_inter
        l_heading = self._heading_loss(pred_wp, gt_wp)
        l_smooth = self._smoothness_loss(pred_wp)

        total = l_frenet + self.lambda_heading * l_heading + self.lambda_smooth * l_smooth
        return {
            "frenet":  l_frenet,
            "heading": l_heading,
            "smooth":  l_smooth,
            "total":   total,
        }



def compile_fdat_loss(loss_cfg: dict | None = None, device: torch.device | None = None) -> FDATLoss:
    loss_cfg = loss_cfg or {}

    criterion = FDATLoss(
        alpha_lane=float(loss_cfg.get("alpha_lane", 20.0)),
        beta_lane=float(loss_cfg.get("beta_lane", 1.0)),
        alpha_inter=float(loss_cfg.get("alpha_inter", 10.0)),
        beta_inter=float(loss_cfg.get("beta_inter", 3.0)),
        lambda_heading=float(loss_cfg.get("lambda_heading", 2.0)),
        lambda_endpoint=float(loss_cfg.get("lambda_endpoint", 5.0)),
        lambda_smooth=float(loss_cfg.get("lambda_smooth", 0.05)),
        tau_start=float(loss_cfg.get("tau_start", 2.0)),
        tau_end=float(loss_cfg.get("tau_end", 4.0)),
        sl1_beta=float(loss_cfg.get("sl1_beta", 0.02)),
    )

    if device is not None:
        criterion = criterion.to(device)

    logger.INFO("Compiled FDATLoss with config:")
    logger.INFO({k: v for k, v in loss_cfg.items()})
    return criterion

    return target_map
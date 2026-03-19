import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger

logger = Logger(__name__)


class UncertaintyWeightedProbeLoss(nn.Module):
	"""
	Multi-task regression loss with homoscedastic uncertainty weighting.

	Default task-to-loss mapping:
	- velocity      -> Log-Cosh
	- steer         -> MAE
	- lateral_error -> MAE

	Objective per enabled task i:
		exp(-s_i) * L_i + s_i
	where s_i is a learnable log-variance parameter.
	"""

	DEFAULT_TASK_ORDER = ("velocity", "steer", "lateral_error")
	DEFAULT_TASK_INDEX = {
		"velocity": 0,
		"steer": 1,
		"lateral_error": 2,
	}
	SUPPORTED_LOSSES = {"log_cosh", "mae", "mse", "smooth_l1"}

	def __init__(
		self,
		enabled: dict[str, bool] | None = None,
		task_to_loss: dict[str, str] | None = None,
		task_to_index: dict[str, int] | None = None,
		initial_log_vars: dict[str, float] | None = None,
		reduction: str = "mean",
	):
		super().__init__()

		if reduction not in {"mean", "sum"}:
			raise ValueError(f"Unsupported reduction '{reduction}'. Use 'mean' or 'sum'.")

		enabled = enabled or {
			"velocity": True,
			"steer": True,
			"lateral_error": True,
		}

		task_to_loss = task_to_loss or {
			"velocity": "log_cosh",
			"steer": "mae",
			"lateral_error": "mae",
		}

		task_to_index = task_to_index or dict(self.DEFAULT_TASK_INDEX)
		initial_log_vars = initial_log_vars or {}

		self.reduction = reduction
		self.task_to_index = dict(task_to_index)
		self.task_to_loss = dict(task_to_loss)

		self.enabled_tasks = [
			task for task in self.DEFAULT_TASK_ORDER if bool(enabled.get(task, False))
		]
		if len(self.enabled_tasks) == 0:
			raise ValueError("At least one task must be enabled for UncertaintyWeightedProbeLoss.")

		for task in self.enabled_tasks:
			loss_name = self.task_to_loss.get(task)
			if loss_name not in self.SUPPORTED_LOSSES:
				raise ValueError(
					f"Unsupported loss '{loss_name}' for task '{task}'. "
					f"Supported: {sorted(self.SUPPORTED_LOSSES)}"
				)

		self.log_vars = nn.ParameterDict(
			{
				task: nn.Parameter(
					torch.tensor(float(initial_log_vars.get(task, 0.0)), dtype=torch.float32)
				)
				for task in self.enabled_tasks
			}
		)

		logger.INFO("Enabled uncertainty-weighted tasks:", self.enabled_tasks)
		logger.INFO("Task losses:", {task: self.task_to_loss[task] for task in self.enabled_tasks})

	@staticmethod
	def _log_cosh(residual: torch.Tensor) -> torch.Tensor:
		# numerically stable log(cosh(x))
		return residual + F.softplus(-2.0 * residual) - math.log(2.0)

	def _base_loss(self, task: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		loss_name = self.task_to_loss[task]
		if loss_name == "log_cosh":
			value = self._log_cosh(pred - target)
		elif loss_name == "mae":
			value = torch.abs(pred - target)
		elif loss_name == "mse":
			value = (pred - target) ** 2
		elif loss_name == "smooth_l1":
			value = F.smooth_l1_loss(pred, target, reduction="none")
		else:
			raise RuntimeError(f"Unexpected loss type '{loss_name}'")

		if self.reduction == "mean":
			return value.mean()
		return value.sum()

	def _get_task_tensor(
		self,
		tensor_or_map: torch.Tensor | dict[str, torch.Tensor],
		task: str,
	) -> torch.Tensor:
		if isinstance(tensor_or_map, dict):
			if task not in tensor_or_map:
				raise KeyError(f"Missing key '{task}' in mapping input.")
			return tensor_or_map[task]

		idx = self.task_to_index.get(task)
		if idx is None:
			raise KeyError(f"Missing index mapping for task '{task}'.")
		return tensor_or_map[..., idx]

	def forward(
		self,
		prediction: torch.Tensor | dict[str, torch.Tensor],
		target: torch.Tensor | dict[str, torch.Tensor],
	) -> tuple[torch.Tensor, dict[str, Any]]:
		total = torch.zeros((), device=next(self.parameters()).device)
		details: dict[str, Any] = {
			"enabled_tasks": list(self.enabled_tasks),
			"per_task": {},
		}

		for task in self.enabled_tasks:
			pred_t = self._get_task_tensor(prediction, task)
			tgt_t = self._get_task_tensor(target, task)

			base = self._base_loss(task, pred_t, tgt_t)
			log_var = self.log_vars[task]
			weighted = torch.exp(-log_var) * base + log_var

			total = total + weighted
			details["per_task"][task] = {
				"loss_type": self.task_to_loss[task],
				"base_loss": base.detach(),
				"weighted_loss": weighted.detach(),
				"log_var": log_var.detach(),
				"weight": torch.exp(-log_var.detach()),
			}

		details["total_loss"] = total.detach()
		return total, details


def compile_loss(loss_cfg: dict | None = None, device: torch.device | None = None) -> UncertaintyWeightedProbeLoss:
	loss_cfg = loss_cfg or {}

	enabled = {
		"velocity": bool(loss_cfg.get("enable_velocity", True)),
		"steer": bool(loss_cfg.get("enable_steer", True)),
		"lateral_error": bool(loss_cfg.get("enable_lateral_error", True)),
	}

	task_to_loss = {
		"velocity": loss_cfg.get("velocity_loss", "log_cosh"),
		"steer": loss_cfg.get("steer_loss", "mae"),
		"lateral_error": loss_cfg.get("lateral_error_loss", "mae"),
	}

	task_to_index = {
		"velocity": int(loss_cfg.get("velocity_idx", 0)),
		"steer": int(loss_cfg.get("steer_idx", 1)),
		"lateral_error": int(loss_cfg.get("lateral_error_idx", 2)),
	}

	initial_log_vars = {
		"velocity": float(loss_cfg.get("velocity_log_var", 0.0)),
		"steer": float(loss_cfg.get("steer_log_var", 0.0)),
		"lateral_error": float(loss_cfg.get("lateral_error_log_var", 0.0)),
	}

	criterion = UncertaintyWeightedProbeLoss(
		enabled=enabled,
		task_to_loss=task_to_loss,
		task_to_index=task_to_index,
		initial_log_vars=initial_log_vars,
		reduction=loss_cfg.get("reduction", "mean"),
	)

	if device is not None:
		criterion = criterion.to(device)

	return criterion


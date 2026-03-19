import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Prober(nn.Module):
    def __init__(self, output_dim: int = 1, init_scales=None, init_shifts=None):
        super().__init__()
        self.output_dim = output_dim

        scales = self._normalize_init_values(init_scales, output_dim, default=1.0)
        shifts = self._normalize_init_values(init_shifts, output_dim, default=0.0)

        if len(scales) != output_dim or len(shifts) != output_dim:
            raise ValueError(
                f"Expected init_scales/init_shifts to match output_dim={output_dim}, "
                f"got len(scales)={len(scales)}, len(shifts)={len(shifts)}"
            )

        self.scale = nn.Parameter(torch.tensor(scales, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(shifts, dtype=torch.float32))

    @staticmethod
    def _normalize_init_values(values, output_dim: int, default: float):
        if values is None:
            values = [default] * output_dim
        if not isinstance(values, (list, tuple, torch.Tensor, np.ndarray)):
            values = [values] * output_dim
        return values

    def apply_output_affine(self, output: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise ValueError("Received empty output sequence from probe heads")
            output = torch.cat(output, dim=-1)

        return (output * F.softplus(self.scale)) + self.shift

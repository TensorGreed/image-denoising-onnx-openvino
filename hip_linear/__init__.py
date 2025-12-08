# hip_linear/__init__.py
import torch
from . import hip_linear_ext


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    hipBLAS-backed linear: y = x @ W^T + b

    x: [B, K]
    weight: [N, K]
    bias: [N]
    """
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("All tensors must be on CUDA/ROCm device for hip_linear.linear")

    return hip_linear_ext.linear_forward(x, weight, bias)

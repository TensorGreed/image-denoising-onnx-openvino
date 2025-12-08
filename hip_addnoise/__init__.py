# hip_addnoise/__init__.py
import torch
from . import hip_addnoise_ext


def add_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Add Gaussian noise on GPU via HIP kernel.

    Args:
        x: [B, 1, 28, 28] or any float32 CUDA/ROCm tensor
        noise_std: standard deviation of noise

    Returns:
        noisy tensor with same shape/dtype as x
    """
    if not x.is_cuda:
        raise ValueError("Input to hip_addnoise.add_noise must be on CUDA/ROCm device")

    noise = torch.randn_like(x)
    return hip_addnoise_ext.add_noise_forward(x, noise, float(noise_std))

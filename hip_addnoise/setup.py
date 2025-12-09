# hip_addnoise/setup.py
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Ensure CUDA_HOME points to ROCm so torch extension build finds hipcc libs.
if "CUDA_HOME" not in os.environ:
    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
    os.environ["CUDA_HOME"] = rocm_home

setup(
    name="hip_addnoise",
    ext_modules=[
        CUDAExtension(
            name="hip_addnoise_ext",
            sources=["add_noise_hip.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": []  # On ROCm, this maps to hipcc.
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

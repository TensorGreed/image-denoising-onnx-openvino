# hip_addnoise/setup.py
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Ensure CUDA_HOME/CUDACXX point to ROCm so torch extension build finds hipcc.
rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
if "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = rocm_home
hipcc = os.path.join(rocm_home, "bin", "hipcc")
if "CUDACXX" not in os.environ and os.path.exists(hipcc):
    os.environ["CUDACXX"] = hipcc

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

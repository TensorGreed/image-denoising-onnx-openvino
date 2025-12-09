# hip_linear/setup.py
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
    name="hip_linear",
    ext_modules=[
        CUDAExtension(
            name="hip_linear_ext",
            sources=["linear_hipblas.cpp"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": []  # On ROCm this is routed to hipcc
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

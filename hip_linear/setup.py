# hip_linear/setup.py
import os
import stat
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Ensure CUDA_HOME/CUDACXX point to ROCm so torch extension build finds hipcc.
rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
hipcc = os.path.join(rocm_home, "bin", "hipcc")
shim_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "rocm_nvcc_shim"))
shim_bin = os.path.join(shim_root, "bin")
shim_nvcc = os.path.join(shim_bin, "nvcc")

os.makedirs(shim_bin, exist_ok=True)

# Create an nvcc shim that forwards to hipcc if it doesn't already exist.
if not os.path.exists(shim_nvcc) and os.path.exists(hipcc):
    with open(shim_nvcc, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n\"{}\" \"$@\"\n".format(hipcc))
    st = os.stat(shim_nvcc)
    os.chmod(shim_nvcc, st.st_mode | stat.S_IEXEC)

# Point CUDA_HOME to the shim so torch's CUDAExtension finds an nvcc executable.
os.environ["CUDA_HOME"] = shim_root
# Point CUDACXX to hipcc directly.
if os.path.exists(hipcc):
    os.environ["CUDACXX"] = hipcc
    os.environ["NVCC"] = hipcc
# Ensure shim bin is on PATH
os.environ["PATH"] = shim_bin + os.pathsep + os.environ.get("PATH", "")

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

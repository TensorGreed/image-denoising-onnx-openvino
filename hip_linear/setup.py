# hip_linear/setup.py
import os
import stat
from setuptools import setup

# --- Set env and shim before importing torch.cpp_extension ---
rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
hipcc = os.path.join(rocm_home, "bin", "hipcc")
shim_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "rocm_nvcc_shim"))
shim_bin = os.path.join(shim_root, "bin")
shim_nvcc = os.path.join(shim_bin, "nvcc")
shim_include = os.path.join(shim_root, "include")
shim_lib64 = os.path.join(shim_root, "lib64")

os.makedirs(shim_bin, exist_ok=True)

# Create an nvcc shim that forwards to hipcc if it doesn't already exist.
if not os.path.exists(shim_nvcc) and os.path.exists(hipcc):
    with open(shim_nvcc, "w", encoding="utf-8") as f:
        f.write(
            "#!/usr/bin/env bash\n"
            "if [[ \"$1\" == \"--version\" ]]; then\n"
            "  echo \"Cuda compilation tools, release 12.0, V12.0.76\"\n"
            "  exit 0\n"
            "fi\n"
            "\"{}\" \"$@\"\n".format(hipcc)
        )
    st = os.stat(shim_nvcc)
    os.chmod(shim_nvcc, st.st_mode | stat.S_IEXEC)

# Mirror include/lib paths to ROCm so include_paths(cuda=True) resolves.
if os.path.exists(rocm_home):
    if not os.path.exists(shim_include):
        try:
            os.symlink(os.path.join(rocm_home, "include"), shim_include)
        except FileExistsError:
            pass
    for libdir_name in ("lib64", "lib"):
        target_lib = os.path.join(rocm_home, libdir_name)
        if os.path.exists(target_lib):
            if not os.path.exists(shim_lib64):
                try:
                    os.symlink(target_lib, shim_lib64)
                except FileExistsError:
                    pass
            break

# Force torch to see the shim as CUDA_HOME and hipcc as NVCC/CUDACXX
os.environ["CUDA_HOME"] = shim_root
if os.path.exists(hipcc):
    os.environ["CUDACXX"] = hipcc
    os.environ["NVCC"] = hipcc
# Default arch for MI300X (gfx942) if not provided
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "gfx942")
os.environ["PATH"] = shim_bin + os.pathsep + os.environ.get("PATH", "")

# Now import torch extension utilities
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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

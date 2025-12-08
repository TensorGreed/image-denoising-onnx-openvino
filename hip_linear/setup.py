# hip_linear/setup.py
from setuptools import setup
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

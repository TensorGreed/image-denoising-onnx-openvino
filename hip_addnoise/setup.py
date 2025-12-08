# hip_addnoise/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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

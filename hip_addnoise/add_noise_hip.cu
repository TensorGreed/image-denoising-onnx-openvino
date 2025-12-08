// hip_addnoise/add_noise_hip.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

namespace {

__global__ void add_noise_kernel(
    const float* __restrict__ clean,
    const float* __restrict__ noise,
    const float noise_std,
    float* __restrict__ out,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = clean[idx] + noise_std * noise[idx];
    }
}

} // anonymous namespace

torch::Tensor add_noise_forward(
    torch::Tensor clean,
    torch::Tensor noise,
    double noise_std_double
) {
    TORCH_CHECK(clean.is_cuda(), "clean tensor must be on CUDA/ROCm device");
    TORCH_CHECK(noise.is_cuda(), "noise tensor must be on CUDA/ROCm device");
    TORCH_CHECK(clean.scalar_type() == torch::kFloat,
                "clean tensor must be float32");
    TORCH_CHECK(noise.scalar_type() == torch::kFloat,
                "noise tensor must be float32");
    TORCH_CHECK(clean.sizes() == noise.sizes(),
                "clean and noise must have the same shape");

    auto out = torch::empty_like(clean);
    int64_t numel = clean.numel();

    const int threads = 256;
    const int blocks = (int)((numel + threads - 1) / threads);

    float noise_std = static_cast<float>(noise_std_double);

    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    hipLaunchKernelGGL(
        add_noise_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        stream,
        clean.data_ptr<float>(),
        noise.data_ptr<float>(),
        noise_std,
        out.data_ptr<float>(),
        numel
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "add_noise_forward",
        &add_noise_forward,
        "Add Gaussian noise (HIP kernel)",
        py::arg("clean"),
        py::arg("noise"),
        py::arg("noise_std")
    );
}

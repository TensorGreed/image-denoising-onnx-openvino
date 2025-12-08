// hip_addnoise/add_noise_hip.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <hip/hip_runtime.h>
#include <limits>

namespace {

template <typename scalar_t>
__global__ void add_noise_kernel(
    const scalar_t* __restrict__ clean,
    const scalar_t* __restrict__ noise,
    const scalar_t noise_std,
    scalar_t* __restrict__ out,
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
    TORCH_CHECK(
        clean.scalar_type() == torch::kFloat ||
            clean.scalar_type() == torch::kBFloat16 ||
            clean.scalar_type() == torch::kHalf,
        "clean tensor must be float32, bfloat16, or float16");
    TORCH_CHECK(clean.scalar_type() == noise.scalar_type(),
                "clean and noise must have the same dtype");
    TORCH_CHECK(clean.device() == noise.device(),
                "clean and noise must be on the same device");
    TORCH_CHECK(clean.sizes() == noise.sizes(),
                "clean and noise must have the same shape");

    if (!clean.is_contiguous()) {
        clean = clean.contiguous();
    }
    if (!noise.is_contiguous()) {
        noise = noise.contiguous();
    }

    auto out = torch::empty_like(clean);
    const int64_t numel = clean.numel();

    // MI300X prefers wider blocks; 512 is a good default to try first.
    const int threads = 512;
    const int64_t blocks_64 = (numel + threads - 1) / threads;
    TORCH_CHECK(
        blocks_64 <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "Tensor is too large for grid launch: blocks=", blocks_64);
    const uint32_t blocks = static_cast<uint32_t>(blocks_64);

    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::Half, at::BFloat16, clean.scalar_type(), "add_noise_forward", [&] {
            scalar_t noise_std = static_cast<scalar_t>(noise_std_double);
            hipLaunchKernelGGL(
                add_noise_kernel<scalar_t>,
                dim3(blocks),
                dim3(threads),
                0,
                stream,
                clean.data_ptr<scalar_t>(),
                noise.data_ptr<scalar_t>(),
                noise_std,
                out.data_ptr<scalar_t>(),
                numel);
        });
    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "add_noise_kernel launch failed: ", hipGetErrorString(err));

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

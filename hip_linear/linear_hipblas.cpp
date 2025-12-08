// hip_linear/linear_hipblas.cpp
#include <torch/extension.h>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

// y = x * W^T + b
// x: [B, K]
// W: [N, K]  (row-major: N rows, K cols)
// b: [N]
// y: [B, N]
torch::Tensor linear_forward_hipblas(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b)
{
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA/ROCm device");
    TORCH_CHECK(w.is_cuda(), "w must be on CUDA/ROCm device");
    TORCH_CHECK(b.is_cuda(), "b must be on CUDA/ROCm device");

    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == torch::kFloat, "w must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat, "b must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, K]");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N, K]");
    TORCH_CHECK(b.dim() == 1, "b must be 1D [N]");

    const int64_t B = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = w.size(0);

    TORCH_CHECK(w.size(1) == K, "w shape must be [N, K] with K matching x.shape[1]");
    TORCH_CHECK(b.size(0) == N, "b shape must be [N] matching W.shape[0]");

    // Output
    auto y = torch::empty({B, N}, x.options());

    // Make sure everything is contiguous
    x = x.contiguous();
    w = w.contiguous();
    b = b.contiguous();
    y = y.contiguous();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // hipBLAS handle
    hipblasHandle_t handle;
    hipblasCreate(&handle);

    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    hipblasSetStream(handle, stream);

    // We want: y = x * W^T
    // Shapes: x [B,K], W [N,K], y [B,N]
    // In BLAS (column-major) terms, we can compute:
    //   Y^T [N,B] = W [N,K] * X^T [K,B]
    //
    // So:
    //   C = A * B
    //   A = W [N,K] (no transpose)
    //   B = X^T [K,B] (transpose of x)
    //   C = Y^T [N,B]
    //
    // That is GEMM: (N x K) * (K x B) = (N x B)

    const int m = static_cast<int>(N);  // rows of A (W)
    const int n = static_cast<int>(B);  // cols of B (X^T)
    const int k = static_cast<int>(K);  // shared dim

    const int lda = k;  // leading dim of A (W)  [N,K] row-major
    const int ldb = k;  // leading dim of B (X^T) -> we use x with transpose flag
    const int ldc = n;  // leading dim of C (Y^T) [N,B]

    // We'll store Y^T into a temporary and then transpose it into y
    auto yT = torch::empty({N, B}, x.options()).contiguous();

    hipblasOperation_t opA = HIPBLAS_OP_N;  // W as-is
    hipblasOperation_t opB = HIPBLAS_OP_T;  // X^T

    hipblasStatus_t stat = hipblasSgemm(
        handle,
        opA,
        opB,
        m, n, k,
        &alpha,
        w.data_ptr<float>(), lda,
        x.data_ptr<float>(), ldb,
        &beta,
        yT.data_ptr<float>(), ldc
    );

    TORCH_CHECK(stat == HIPBLAS_STATUS_SUCCESS, "hipblasSgemm failed");

    hipblasDestroy(handle);

    // yT is [N,B], row-major. We want y [B,N].
    y.copy_(yT.transpose(0, 1));

    // Add bias
    // y: [B,N], b: [N] -> broadcasts over dim 0
    y.add_(b);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "linear_forward",
        &linear_forward_hipblas,
        "Linear layer (hipBLAS) forward",
        py::arg("x"),
        py::arg("w"),
        py::arg("b")
    );
}

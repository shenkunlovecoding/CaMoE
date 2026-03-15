#include <torch/extension.h>

using torch::Tensor;

Tensor diagonal_scan_forward_cuda(const Tensor& x);
Tensor diagonal_scan_backward_cuda(const Tensor& x, const Tensor& y, const Tensor& grad_output);

Tensor diagonal_scan_forward_dispatch(const Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, T]");
    TORCH_CHECK(x.size(1) == x.size(2), "x must be square over the last two dims");
    return diagonal_scan_forward_cuda(x);
}

Tensor diagonal_scan_backward_dispatch(const Tensor& x, const Tensor& y, const Tensor& grad_output) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && grad_output.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat, "y must be float32");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat, "grad_output must be float32");
    TORCH_CHECK(x.sizes() == y.sizes() && x.sizes() == grad_output.sizes(), "shapes must match");
    TORCH_CHECK(x.dim() == 3 && x.size(1) == x.size(2), "x must have shape [B, T, T]");
    return diagonal_scan_backward_cuda(x, y, grad_output);
}

TORCH_LIBRARY(soft_rosa_scan, m) {
    m.def("forward(Tensor x) -> Tensor");
    m.def("backward(Tensor x, Tensor y, Tensor grad_output) -> Tensor");
}

TORCH_LIBRARY_IMPL(soft_rosa_scan, CUDA, m) {
    m.impl("forward", &diagonal_scan_forward_dispatch);
    m.impl("backward", &diagonal_scan_backward_dispatch);
}

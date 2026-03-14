#include <torch/extension.h>

using torch::Tensor;

std::tuple<Tensor, Tensor, Tensor> qkv1bit_forward_cuda(const Tensor& q, const Tensor& k, const Tensor& v, int64_t K);

std::tuple<Tensor, Tensor, Tensor> qkv1bit_forward_dispatch(const Tensor& q, const Tensor& k, const Tensor& v, int64_t K) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kUInt8, "q must be uint8");
    TORCH_CHECK(k.dtype() == torch::kUInt8, "k must be uint8");
    TORCH_CHECK(v.dtype() == torch::kUInt8, "v must be uint8");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "q/k/v must have shape [S, T]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shapes must match");
    TORCH_CHECK(K >= 1, "K must be >= 1");
    return qkv1bit_forward_cuda(q, k, v, K);
}

TORCH_LIBRARY(soft_rosa_qkv1bit, m) {
    m.def("forward(Tensor q, Tensor k, Tensor v, int K) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(soft_rosa_qkv1bit, CUDA, m) {
    m.impl("forward", &qkv1bit_forward_dispatch);
}

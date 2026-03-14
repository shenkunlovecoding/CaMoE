#include <torch/extension.h>
using torch::Tensor;
using uchar = unsigned char;

void cuda_forward(int B, int H, uchar*q, uchar*k, uchar*v, uchar*y, uchar*scratch);
void cuda_forward_with_match_len(int B, int H, uchar*q, uchar*k, uchar*v, uchar*y, int*match_len, uchar*scratch);
void cuda_backward(int B, int H, uchar*q, uchar*k, uchar*v, float*dy, float*dq, float*dk, float*dv, uchar*scratch);
int64_t scratch_size(int64_t B, int64_t H);

void cuda_forward_(Tensor &q, Tensor &k, Tensor &v, Tensor &y, Tensor &scratch) {
    int B = q.size(0), H = q.size(1);
    cuda_forward(B, H, (uchar*)q.data_ptr(), (uchar*)k.data_ptr(), (uchar*)v.data_ptr(), (uchar*)y.data_ptr(), (uchar*)scratch.data_ptr());
}
void cuda_forward_with_match_len_(Tensor &q, Tensor &k, Tensor &v, Tensor &y, Tensor &match_len, Tensor &scratch) {
    int B = q.size(0), H = q.size(1);
    cuda_forward_with_match_len(
        B,
        H,
        (uchar*)q.data_ptr(),
        (uchar*)k.data_ptr(),
        (uchar*)v.data_ptr(),
        (uchar*)y.data_ptr(),
        (int*)match_len.data_ptr(),
        (uchar*)scratch.data_ptr()
    );
}
void cuda_backward_(Tensor &q, Tensor &k, Tensor &v, Tensor &dy, Tensor &dq, Tensor &dk, Tensor &dv, Tensor &scratch) {
    int B = q.size(0), H = q.size(1);
    cuda_backward(B, H, (uchar*)q.data_ptr(), (uchar*)k.data_ptr(), (uchar*)v.data_ptr(), (float*)dy.data_ptr(), (float*)dq.data_ptr(), (float*)dk.data_ptr(), (float*)dv.data_ptr(), (uchar*)scratch.data_ptr());
}

TORCH_LIBRARY(wind_rosa, m) {
    m.def("forward(Tensor q, Tensor k, Tensor v, Tensor(y!) y, Tensor(s!) scratch) -> ()");
    m.def("forward_with_match_len(Tensor q, Tensor k, Tensor v, Tensor(y!) y, Tensor(m!) match_len, Tensor(s!) scratch) -> ()");
    m.def("backward(Tensor q, Tensor k, Tensor v, Tensor dy, Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv, Tensor(s!) scratch) -> ()");
    m.def("scratch_size(int B, int H) -> int");
    m.impl("scratch_size", &scratch_size);
}

TORCH_LIBRARY_IMPL(wind_rosa, CUDA, m) {
    m.impl("forward", &cuda_forward_);
    m.impl("forward_with_match_len", &cuda_forward_with_match_len_);
    m.impl("backward", &cuda_backward_);
}

#include <torch/extension.h>

using torch::Tensor;
using uchar = unsigned char;

#ifndef ROSA_SUFA_TRUNCATED_FORWARD_OP
#define ROSA_SUFA_TRUNCATED_FORWARD_OP forward_with_metadata
#endif

#ifndef ROSA_SUFA_TRUNCATED_BACKWARD_OP
#define ROSA_SUFA_TRUNCATED_BACKWARD_OP backward
#endif

#ifndef ROSA_SUFA_TRUNCATED_SCRATCH_OP
#define ROSA_SUFA_TRUNCATED_SCRATCH_OP scratch_size
#endif

#define ROSA_SUFA_STRINGIFY_IMPL(x) #x
#define ROSA_SUFA_STRINGIFY(x) ROSA_SUFA_STRINGIFY_IMPL(x)

#define ROSA_SUFA_FORWARD_SCHEMA \
    ROSA_SUFA_STRINGIFY(ROSA_SUFA_TRUNCATED_FORWARD_OP) \
    "(Tensor q, Tensor k, Tensor v, Tensor(y!) y, Tensor(m!) match_len, Tensor(e!) endpos, Tensor(s!) scratch) -> ()"

#define ROSA_SUFA_BACKWARD_SCHEMA \
    ROSA_SUFA_STRINGIFY(ROSA_SUFA_TRUNCATED_BACKWARD_OP) \
    "(Tensor q, Tensor k, Tensor v, Tensor dy, Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv, Tensor(s!) scratch) -> ()"

#define ROSA_SUFA_SCRATCH_SCHEMA \
    ROSA_SUFA_STRINGIFY(ROSA_SUFA_TRUNCATED_SCRATCH_OP) \
    "(int B, int H) -> int"

#define ROSA_SUFA_FORWARD_NAME ROSA_SUFA_STRINGIFY(ROSA_SUFA_TRUNCATED_FORWARD_OP)
#define ROSA_SUFA_BACKWARD_NAME ROSA_SUFA_STRINGIFY(ROSA_SUFA_TRUNCATED_BACKWARD_OP)
#define ROSA_SUFA_SCRATCH_NAME ROSA_SUFA_STRINGIFY(ROSA_SUFA_TRUNCATED_SCRATCH_OP)

void cuda_forward_with_metadata(
    int B,
    int H,
    uchar* q,
    uchar* k,
    uchar* v,
    uchar* y,
    int* match_len,
    int* endpos,
    uchar* scratch
);
void cuda_backward(int B, int H, uchar* q, uchar* k, uchar* v, float* dy, float* dq, float* dk, float* dv, uchar* scratch);
int64_t scratch_size(int64_t B, int64_t H);

void cuda_forward_with_metadata_(
    Tensor& q,
    Tensor& k,
    Tensor& v,
    Tensor& y,
    Tensor& match_len,
    Tensor& endpos,
    Tensor& scratch
) {
    int B = q.size(0), H = q.size(1);
    cuda_forward_with_metadata(
        B,
        H,
        (uchar*)q.data_ptr(),
        (uchar*)k.data_ptr(),
        (uchar*)v.data_ptr(),
        (uchar*)y.data_ptr(),
        (int*)match_len.data_ptr(),
        (int*)endpos.data_ptr(),
        (uchar*)scratch.data_ptr()
    );
}

void cuda_backward_(
    Tensor& q,
    Tensor& k,
    Tensor& v,
    Tensor& dy,
    Tensor& dq,
    Tensor& dk,
    Tensor& dv,
    Tensor& scratch
) {
    int B = q.size(0), H = q.size(1);
    cuda_backward(
        B,
        H,
        (uchar*)q.data_ptr(),
        (uchar*)k.data_ptr(),
        (uchar*)v.data_ptr(),
        (float*)dy.data_ptr(),
        (float*)dq.data_ptr(),
        (float*)dk.data_ptr(),
        (float*)dv.data_ptr(),
        (uchar*)scratch.data_ptr()
    );
}

TORCH_LIBRARY_FRAGMENT(rosa_sufa_truncated, m) {
    m.def(ROSA_SUFA_FORWARD_SCHEMA);
    m.def(ROSA_SUFA_BACKWARD_SCHEMA);
    m.def(ROSA_SUFA_SCRATCH_SCHEMA);
    m.impl(ROSA_SUFA_SCRATCH_NAME, &scratch_size);
}

TORCH_LIBRARY_IMPL(rosa_sufa_truncated, CUDA, m) {
    m.impl(ROSA_SUFA_FORWARD_NAME, &cuda_forward_with_metadata_);
    m.impl(ROSA_SUFA_BACKWARD_NAME, &cuda_backward_);
}

#include <ATen/ATen.h>

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

using at::Tensor;
using uchar = unsigned char;

__global__ void qkv1bit_forward_kernel(
    const uchar* q,
    const uchar* k,
    const uchar* v,
    uchar* y,
    int64_t* best_j,
    int64_t* best_len,
    int num_streams,
    int seq_len,
    int K
) {
    int s = blockIdx.x;
    if (s >= num_streams) {
        return;
    }

    const uchar* q_s = q + static_cast<size_t>(s) * seq_len;
    const uchar* k_s = k + static_cast<size_t>(s) * seq_len;
    const uchar* v_s = v + static_cast<size_t>(s) * seq_len;
    uchar* y_s = y + static_cast<size_t>(s) * seq_len;
    int64_t* bj_s = best_j + static_cast<size_t>(s) * seq_len;
    int64_t* bl_s = best_len + static_cast<size_t>(s) * seq_len;

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        int cur_best_len = 0;
        int cur_best_j = -1;
        int max_check = min(K, i + 1);

        for (int j = 0; j < i; ++j) {
            int max_match = min(max_check, j + 1);
            int len = 0;

            while (len < max_match && q_s[i - len] == k_s[j - len]) {
                ++len;
            }

            if (len > cur_best_len || (len == cur_best_len && len > 0 && j > cur_best_j)) {
                cur_best_len = len;
                cur_best_j = j;
            }
        }

        bj_s[i] = cur_best_j;
        bl_s[i] = cur_best_len;
        y_s[i] = cur_best_j >= 0 ? v_s[cur_best_j + 1] : static_cast<uchar>(0);
    }
}

std::tuple<Tensor, Tensor, Tensor> qkv1bit_forward_cuda(const Tensor& q, const Tensor& k, const Tensor& v, int64_t K) {
    auto output = at::zeros_like(v);
    auto best_j = at::full({q.size(0), q.size(1)}, -1, q.options().dtype(at::kLong));
    auto best_len = at::zeros({q.size(0), q.size(1)}, q.options().dtype(at::kLong));

    int num_streams = static_cast<int>(q.size(0));
    int seq_len = static_cast<int>(q.size(1));
    int threads = 1;
    while (threads < seq_len && threads < 256) {
        threads <<= 1;
    }

    qkv1bit_forward_kernel<<<num_streams, threads>>>(
        q.data_ptr<uchar>(),
        k.data_ptr<uchar>(),
        v.data_ptr<uchar>(),
        output.data_ptr<uchar>(),
        best_j.data_ptr<int64_t>(),
        best_len.data_ptr<int64_t>(),
        num_streams,
        seq_len,
        static_cast<int>(K)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {output, best_j, best_len};
}

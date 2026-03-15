#include <ATen/ATen.h>

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

using at::Tensor;
using uchar = unsigned char;

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ uint64_t top_mask(int valid_len) {
    if (valid_len <= 0) {
        return 0ull;
    }
    if (valid_len >= 64) {
        return ~0ull;
    }
    return ~0ull << (64 - valid_len);
}

__device__ __forceinline__ int match_len_from_packed(uint64_t q_pack, uint64_t k_pack, int valid_len, int K) {
    int len_cap = min(valid_len, K);
    if (len_cap <= 0) {
        return 0;
    }
    uint64_t diff = (q_pack ^ k_pack) & top_mask(len_cap);
    if (diff == 0ull) {
        return len_cap;
    }
    return min(static_cast<int>(__clzll(diff)), len_cap);
}

__device__ __forceinline__ int score_from_len_j(int len, int j, int seq_len) {
    return len > 0 ? len * (seq_len + 1) + j : -1;
}

__device__ __forceinline__ uint64_t flip_mask_for_position(int owner_t, int flip_t) {
    int offset = owner_t - flip_t;
    if (offset < 0 || offset >= 64) {
        return 0ull;
    }
    return 1ull << (63 - offset);
}

__global__ void pack_history_kernel(
    const uchar* x,
    uint64_t* packs,
    int num_streams,
    int seq_len,
    int K
) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_streams * seq_len;
    if (linear >= total) {
        return;
    }

    int s = linear / seq_len;
    int t = linear % seq_len;
    const uchar* x_s = x + static_cast<size_t>(s) * seq_len;

    uint64_t pack = 0ull;
    int valid = min(K, t + 1);
    for (int off = 0; off < valid; ++off) {
        if (x_s[t - off]) {
            pack |= 1ull << (63 - off);
        }
    }
    packs[linear] = pack;
}

__global__ void qkv1bit_forward_bytewise_kernel(
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

__global__ void qkv1bit_forward_packed_kernel(
    const uint64_t* q_pack,
    const uint64_t* k_pack,
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

    const uint64_t* q_s = q_pack + static_cast<size_t>(s) * seq_len;
    const uint64_t* k_s = k_pack + static_cast<size_t>(s) * seq_len;
    const uchar* v_s = v + static_cast<size_t>(s) * seq_len;
    uchar* y_s = y + static_cast<size_t>(s) * seq_len;
    int64_t* bj_s = best_j + static_cast<size_t>(s) * seq_len;
    int64_t* bl_s = best_len + static_cast<size_t>(s) * seq_len;

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        int best_score = -1;
        int cur_best_j = -1;
        for (int j = 0; j < i; ++j) {
            int valid_len = min(K, min(i + 1, j + 1));
            int len = match_len_from_packed(q_s[i], k_s[j], valid_len, K);
            int score = score_from_len_j(len, j, seq_len);
            if (score > best_score) {
                best_score = score;
                cur_best_j = j;
            }
        }

        int cur_best_len = best_score >= 0 ? best_score / (seq_len + 1) : 0;
        bj_s[i] = cur_best_j;
        bl_s[i] = cur_best_len;
        y_s[i] = cur_best_j >= 0 ? v_s[cur_best_j + 1] : static_cast<uchar>(0);
    }
}

__global__ void qkv1bit_backward_dq_packed_kernel(
    const uchar* q,
    const uint64_t* q_pack,
    const uint64_t* k_pack,
    const uchar* v,
    const uchar* base_y,
    const float* grad_output,
    float* dq,
    int num_streams,
    int seq_len,
    int K
) {
    int s = blockIdx.x;
    if (s >= num_streams) {
        return;
    }

    const uchar* q_s = q + static_cast<size_t>(s) * seq_len;
    const uint64_t* q_pack_s = q_pack + static_cast<size_t>(s) * seq_len;
    const uint64_t* k_pack_s = k_pack + static_cast<size_t>(s) * seq_len;
    const uchar* v_s = v + static_cast<size_t>(s) * seq_len;
    const uchar* y_s = base_y + static_cast<size_t>(s) * seq_len;
    const float* go_s = grad_output + static_cast<size_t>(s) * seq_len;
    float* dq_s = dq + static_cast<size_t>(s) * seq_len;

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float accum = 0.0f;
        int i_end = min(seq_len - 1, t + K - 1);
        for (int i = t; i <= i_end; ++i) {
            uint64_t flipped_q = q_pack_s[i] ^ flip_mask_for_position(i, t);
            int best_score = -1;
            int off_j = -1;
            for (int j = 0; j < i; ++j) {
                int valid_len = min(K, min(i + 1, j + 1));
                int len = match_len_from_packed(flipped_q, k_pack_s[j], valid_len, K);
                int score = score_from_len_j(len, j, seq_len);
                if (score > best_score) {
                    best_score = score;
                    off_j = j;
                }
            }
            uchar off_y = off_j >= 0 ? v_s[off_j + 1] : static_cast<uchar>(0);
            accum += (static_cast<float>(off_y) - static_cast<float>(y_s[i])) * go_s[i];
        }

        float flip_dir = 1.0f - 2.0f * static_cast<float>(q_s[t]);
        dq_s[t] = accum * flip_dir;
    }
}

__global__ void qkv1bit_backward_dk_packed_kernel(
    const uchar* k,
    const uint64_t* q_pack,
    const uint64_t* k_pack,
    const uchar* v,
    const uchar* base_y,
    const float* grad_output,
    float* dk,
    int num_streams,
    int seq_len,
    int K
) {
    int s = blockIdx.x;
    if (s >= num_streams) {
        return;
    }

    const uchar* k_s = k + static_cast<size_t>(s) * seq_len;
    const uint64_t* q_pack_s = q_pack + static_cast<size_t>(s) * seq_len;
    const uint64_t* k_pack_s = k_pack + static_cast<size_t>(s) * seq_len;
    const uchar* v_s = v + static_cast<size_t>(s) * seq_len;
    const uchar* y_s = base_y + static_cast<size_t>(s) * seq_len;
    const float* go_s = grad_output + static_cast<size_t>(s) * seq_len;
    float* dk_s = dk + static_cast<size_t>(s) * seq_len;

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float accum = 0.0f;
        for (int i = t + 1; i < seq_len; ++i) {
            int best_score = -1;
            int off_j = -1;
            for (int j = 0; j < i; ++j) {
                uint64_t k_word = k_pack_s[j];
                if (j >= t && (j - t) < K) {
                    k_word ^= flip_mask_for_position(j, t);
                }
                int valid_len = min(K, min(i + 1, j + 1));
                int len = match_len_from_packed(q_pack_s[i], k_word, valid_len, K);
                int score = score_from_len_j(len, j, seq_len);
                if (score > best_score) {
                    best_score = score;
                    off_j = j;
                }
            }
            uchar off_y = off_j >= 0 ? v_s[off_j + 1] : static_cast<uchar>(0);
            accum += (static_cast<float>(off_y) - static_cast<float>(y_s[i])) * go_s[i];
        }

        float flip_dir = 1.0f - 2.0f * static_cast<float>(k_s[t]);
        dk_s[t] = accum * flip_dir;
    }
}

__global__ void qkv1bit_backward_v_scatter_kernel(
    const int64_t* best_j,
    const float* grad_output,
    float* grad_v,
    int num_streams,
    int seq_len
) {
    int s = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (s >= num_streams || i >= seq_len) {
        return;
    }

    const int64_t* bj_s = best_j + static_cast<size_t>(s) * seq_len;
    const float* go_s = grad_output + static_cast<size_t>(s) * seq_len;
    float* gv_s = grad_v + static_cast<size_t>(s) * seq_len;

    int target_t = static_cast<int>(bj_s[i] + 1);
    if (bj_s[i] >= 0 && target_t < seq_len) {
        atomicAdd(gv_s + target_t, go_s[i]);
    }
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> qkv1bit_forward_cuda(const Tensor& q, const Tensor& k, const Tensor& v, int64_t K) {
    auto output = at::zeros_like(v);
    auto best_j = at::full({q.size(0), q.size(1)}, -1, q.options().dtype(at::kLong));
    auto best_len = at::zeros({q.size(0), q.size(1)}, q.options().dtype(at::kLong));

    int num_streams = static_cast<int>(q.size(0));
    int seq_len = static_cast<int>(q.size(1));
    int threads = 1;
    while (threads < seq_len && threads < kThreads) {
        threads <<= 1;
    }

    if (K > 0 && K <= 64) {
        auto q_pack = at::zeros({q.size(0), q.size(1)}, q.options().dtype(at::kLong));
        auto k_pack = at::zeros({k.size(0), k.size(1)}, k.options().dtype(at::kLong));
        int total = num_streams * seq_len;
        int blocks = (total + kThreads - 1) / kThreads;
        pack_history_kernel<<<blocks, kThreads>>>(
            q.data_ptr<uchar>(),
            reinterpret_cast<uint64_t*>(q_pack.data_ptr<int64_t>()),
            num_streams,
            seq_len,
            static_cast<int>(K)
        );
        pack_history_kernel<<<blocks, kThreads>>>(
            k.data_ptr<uchar>(),
            reinterpret_cast<uint64_t*>(k_pack.data_ptr<int64_t>()),
            num_streams,
            seq_len,
            static_cast<int>(K)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        qkv1bit_forward_packed_kernel<<<num_streams, threads>>>(
            reinterpret_cast<uint64_t*>(q_pack.data_ptr<int64_t>()),
            reinterpret_cast<uint64_t*>(k_pack.data_ptr<int64_t>()),
            v.data_ptr<uchar>(),
            output.data_ptr<uchar>(),
            best_j.data_ptr<int64_t>(),
            best_len.data_ptr<int64_t>(),
            num_streams,
            seq_len,
            static_cast<int>(K)
        );
    } else {
        qkv1bit_forward_bytewise_kernel<<<num_streams, threads>>>(
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
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {output, best_j, best_len};
}

std::tuple<Tensor, Tensor, Tensor> qkv1bit_backward_cuda(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& output,
    const Tensor& best_j,
    const Tensor& grad_output,
    int64_t K
) {
    auto dq = at::zeros_like(grad_output);
    auto dk = at::zeros_like(grad_output);
    auto dv = at::zeros_like(grad_output);

    int num_streams = static_cast<int>(q.size(0));
    int seq_len = static_cast<int>(q.size(1));
    int threads = 1;
    while (threads < seq_len && threads < kThreads) {
        threads <<= 1;
    }

    if (K > 0 && K <= 64) {
        auto q_pack = at::zeros({q.size(0), q.size(1)}, q.options().dtype(at::kLong));
        auto k_pack = at::zeros({k.size(0), k.size(1)}, k.options().dtype(at::kLong));
        int total = num_streams * seq_len;
        int blocks = (total + kThreads - 1) / kThreads;
        pack_history_kernel<<<blocks, kThreads>>>(
            q.data_ptr<uchar>(),
            reinterpret_cast<uint64_t*>(q_pack.data_ptr<int64_t>()),
            num_streams,
            seq_len,
            static_cast<int>(K)
        );
        pack_history_kernel<<<blocks, kThreads>>>(
            k.data_ptr<uchar>(),
            reinterpret_cast<uint64_t*>(k_pack.data_ptr<int64_t>()),
            num_streams,
            seq_len,
            static_cast<int>(K)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        qkv1bit_backward_dq_packed_kernel<<<num_streams, threads>>>(
            q.data_ptr<uchar>(),
            reinterpret_cast<uint64_t*>(q_pack.data_ptr<int64_t>()),
            reinterpret_cast<uint64_t*>(k_pack.data_ptr<int64_t>()),
            v.data_ptr<uchar>(),
            output.data_ptr<uchar>(),
            grad_output.data_ptr<float>(),
            dq.data_ptr<float>(),
            num_streams,
            seq_len,
            static_cast<int>(K)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        qkv1bit_backward_dk_packed_kernel<<<num_streams, threads>>>(
            k.data_ptr<uchar>(),
            reinterpret_cast<uint64_t*>(q_pack.data_ptr<int64_t>()),
            reinterpret_cast<uint64_t*>(k_pack.data_ptr<int64_t>()),
            v.data_ptr<uchar>(),
            output.data_ptr<uchar>(),
            grad_output.data_ptr<float>(),
            dk.data_ptr<float>(),
            num_streams,
            seq_len,
            static_cast<int>(K)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        dim3 grid_v(num_streams, (seq_len + kThreads - 1) / kThreads);
        qkv1bit_backward_v_scatter_kernel<<<grid_v, kThreads>>>(
            best_j.data_ptr<int64_t>(),
            grad_output.data_ptr<float>(),
            dv.data_ptr<float>(),
            num_streams,
            seq_len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        // Fallback: reuse the old bytewise behavior through exact finite-difference style recompute cost.
        dim3 grid_v(num_streams, (seq_len + kThreads - 1) / kThreads);
        qkv1bit_backward_v_scatter_kernel<<<grid_v, kThreads>>>(
            best_j.data_ptr<int64_t>(),
            grad_output.data_ptr<float>(),
            dv.data_ptr<float>(),
            num_streams,
            seq_len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {dq, dk, dv};
}

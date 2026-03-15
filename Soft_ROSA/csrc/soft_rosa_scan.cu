#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

using at::Tensor;

namespace {

struct AffineState {
    float g;
    float a;
};

struct AffineCombine {
    __host__ __device__ __forceinline__ AffineState operator()(const AffineState& lhs, const AffineState& rhs) const {
        return {rhs.g * lhs.g, rhs.g * lhs.a + rhs.a};
    }
};

__global__ void pack_forward_kernel(
    const float* x,
    AffineState* packed,
    int64_t* offsets,
    int64_t* keys,
    int batch,
    int seq_len
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int idx = threadIdx.x + blockDim.x * blockIdx.z;
    if (b >= batch || d >= seq_len) {
        return;
    }

    int diag_len = seq_len - d;
    if (idx >= diag_len) {
        return;
    }

    int64_t seg = static_cast<int64_t>(b) * seq_len + d;
    int64_t base = offsets[seg];
    int r = idx + d;
    int c = idx;
    float v = x[(static_cast<int64_t>(b) * seq_len + r) * seq_len + c];
    packed[base + idx] = {v, v};
    keys[base + idx] = seg;
}

__global__ void unpack_forward_kernel(
    const AffineState* packed,
    const int64_t* offsets,
    float* y,
    int batch,
    int seq_len
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int idx = threadIdx.x + blockDim.x * blockIdx.z;
    if (b >= batch || d >= seq_len) {
        return;
    }

    int diag_len = seq_len - d;
    if (idx >= diag_len) {
        return;
    }

    int64_t seg = static_cast<int64_t>(b) * seq_len + d;
    int64_t base = offsets[seg];
    int r = idx + d;
    int c = idx;
    y[(static_cast<int64_t>(b) * seq_len + r) * seq_len + c] = packed[base + idx].a;
}

__global__ void pack_backward_kernel(
    const float* x,
    const float* grad_output,
    AffineState* packed,
    int64_t* offsets,
    int64_t* keys,
    int batch,
    int seq_len
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int rev_idx = threadIdx.x + blockDim.x * blockIdx.z;
    if (b >= batch || d >= seq_len) {
        return;
    }

    int diag_len = seq_len - d;
    if (rev_idx >= diag_len) {
        return;
    }

    int idx = diag_len - 1 - rev_idx;
    int r = idx + d;
    int c = idx;
    float g = (r < seq_len - 1) ? x[(static_cast<int64_t>(b) * seq_len + (r + 1)) * seq_len + (c + 1)] : 1.0f;
    float a = grad_output[(static_cast<int64_t>(b) * seq_len + r) * seq_len + c];

    int64_t seg = static_cast<int64_t>(b) * seq_len + d;
    int64_t base = offsets[seg];
    packed[base + rev_idx] = {g, a};
    keys[base + rev_idx] = seg;
}

__global__ void unpack_backward_kernel(
    const AffineState* packed,
    const int64_t* offsets,
    const float* y,
    float* grad_input,
    int batch,
    int seq_len
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int rev_idx = threadIdx.x + blockDim.x * blockIdx.z;
    if (b >= batch || d >= seq_len) {
        return;
    }

    int diag_len = seq_len - d;
    if (rev_idx >= diag_len) {
        return;
    }

    int idx = diag_len - 1 - rev_idx;
    int r = idx + d;
    int c = idx;
    float y_prev = idx > 0 ? y[(static_cast<int64_t>(b) * seq_len + (r - 1)) * seq_len + (c - 1)] : 0.0f;

    int64_t seg = static_cast<int64_t>(b) * seq_len + d;
    int64_t base = offsets[seg];
    grad_input[(static_cast<int64_t>(b) * seq_len + r) * seq_len + c] = (y_prev + 1.0f) * packed[base + rev_idx].a;
}

Tensor build_offsets_tensor(int batch, int seq_len, const at::TensorOptions& options) {
    auto offsets_cpu = at::empty({static_cast<int64_t>(batch) * seq_len + 1}, options.device(at::kCPU));
    auto* ptr = offsets_cpu.data_ptr<int64_t>();
    int64_t cur = 0;
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < seq_len; ++d) {
            ptr[static_cast<int64_t>(b) * seq_len + d] = cur;
            cur += seq_len - d;
        }
    }
    ptr[static_cast<int64_t>(batch) * seq_len] = cur;
    return offsets_cpu.to(options.device());
}

struct SegmentEqual {
    __host__ __device__ __forceinline__ bool operator()(const int64_t& lhs, const int64_t& rhs) const {
        return lhs == rhs;
    }
};

void run_segmented_scan(
    const int64_t* keys,
    const AffineState* input,
    AffineState* output,
    int64_t num_items
) {
    size_t temp_storage_bytes = 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveScanByKey(
        nullptr,
        temp_storage_bytes,
        keys,
        input,
        output,
        AffineCombine{},
        num_items,
        SegmentEqual{},
        stream
    ));

    auto temp = at::empty(
        {static_cast<long long>(temp_storage_bytes)},
        at::TensorOptions().device(at::kCUDA).dtype(at::kByte)
    );
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveScanByKey(
        temp.data_ptr(),
        temp_storage_bytes,
        keys,
        input,
        output,
        AffineCombine{},
        num_items,
        SegmentEqual{},
        stream
    ));
}

}  // namespace

Tensor diagonal_scan_forward_cuda(const Tensor& x) {
    int batch = static_cast<int>(x.size(0));
    int seq_len = static_cast<int>(x.size(1));
    auto offsets = build_offsets_tensor(batch, seq_len, x.options().dtype(at::kLong));
    int64_t total_items = static_cast<int64_t>(batch) * seq_len * (seq_len + 1) / 2;
    auto packed_in = at::empty({total_items, 2}, x.options().dtype(at::kFloat));
    auto packed_out = at::empty_like(packed_in);
    auto keys = at::empty({total_items}, x.options().dtype(at::kLong));
    auto y = at::zeros_like(x);

    dim3 block(256);
    dim3 grid(batch, seq_len, (seq_len + block.x - 1) / block.x);
    pack_forward_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        reinterpret_cast<AffineState*>(packed_in.data_ptr<float>()),
        offsets.data_ptr<int64_t>(),
        keys.data_ptr<int64_t>(),
        batch,
        seq_len
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    run_segmented_scan(
        keys.data_ptr<int64_t>(),
        reinterpret_cast<AffineState*>(packed_in.data_ptr<float>()),
        reinterpret_cast<AffineState*>(packed_out.data_ptr<float>()),
        total_items
    );

    unpack_forward_kernel<<<grid, block>>>(
        reinterpret_cast<AffineState*>(packed_out.data_ptr<float>()),
        offsets.data_ptr<int64_t>(),
        y.data_ptr<float>(),
        batch,
        seq_len
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

Tensor diagonal_scan_backward_cuda(const Tensor& x, const Tensor& y, const Tensor& grad_output) {
    int batch = static_cast<int>(x.size(0));
    int seq_len = static_cast<int>(x.size(1));
    auto offsets = build_offsets_tensor(batch, seq_len, x.options().dtype(at::kLong));
    int64_t total_items = static_cast<int64_t>(batch) * seq_len * (seq_len + 1) / 2;
    auto packed_in = at::empty({total_items, 2}, x.options().dtype(at::kFloat));
    auto packed_out = at::empty_like(packed_in);
    auto keys = at::empty({total_items}, x.options().dtype(at::kLong));
    auto grad_input = at::zeros_like(x);

    dim3 block(256);
    dim3 grid(batch, seq_len, (seq_len + block.x - 1) / block.x);
    pack_backward_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        reinterpret_cast<AffineState*>(packed_in.data_ptr<float>()),
        offsets.data_ptr<int64_t>(),
        keys.data_ptr<int64_t>(),
        batch,
        seq_len
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    run_segmented_scan(
        keys.data_ptr<int64_t>(),
        reinterpret_cast<AffineState*>(packed_in.data_ptr<float>()),
        reinterpret_cast<AffineState*>(packed_out.data_ptr<float>()),
        total_items
    );

    unpack_backward_kernel<<<grid, block>>>(
        reinterpret_cast<AffineState*>(packed_out.data_ptr<float>()),
        offsets.data_ptr<int64_t>(),
        y.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        batch,
        seq_len
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
}

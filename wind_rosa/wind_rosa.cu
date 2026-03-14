// Copyright (c) 2026, Johan Sokrates Wind

#include <cub/cub.cuh>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

constexpr int T = _T_, K = _K_, C = _C_, T2 = T*2, R = (C%4 == 0 ? 4 : (C%2 == 0 ? 2 : 1)), CR = C/R;

using uchar = unsigned char;
#if _T_ <= 16384
    using NxtInt = short;
#else
    using NxtInt = int;
#endif

#if _K_*_C_ <= 32
    using KeyInt = uint32_t;
#elif _K_*_C_ <= 64
    using KeyInt = uint64_t;
#else
#error wind_rosa currently requires _K_*_C_ <= 64
#endif

template<int mod>
__global__ void atoi_mod(size_t N, int*ptr) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) ptr[i] = i%mod;
}

template<int S>
__global__ void invert_perm(size_t M, int*perm, int*iperm) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        iperm[i/S*S+perm[i]] = i%S;
    }
}

template<int S, bool Q>
__global__ void extend_key(size_t M, int d, int r, uchar*k, int*perm, KeyInt*key, KeyInt*nkey) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    int t = perm[i], ind = (Q?t%T:t-1)-d;
    nkey[i] = (d < K ? (key[i]<<R | (ind >= 0 ? (KeyInt)(k[i/S*T+ind]>>r*R&((1<<R)-1)) : 0)) : key[i]);
}

template<int S, bool Q>
__global__ void scatter_nxt(size_t M, int d, KeyInt*key, int*perm, int*iperm, NxtInt*nxt) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    int t = perm[i], ind = (Q?t%T:t-1)-d;
    if (ind >= 0) {
        size_t c = key[i]&((1<<R)-1), bi = i/S;
        nxt[bi*((size_t)S<<R) + c*S + iperm[bi*S+t]] = i%S;
    }
}

__device__ int2 query_trie(size_t M, size_t bi, int ti, int qi, uchar*q, NxtInt*nxt, int*time, int flipi = -1, int flipc = -1, int mint = -1) {
    nxt += bi*((size_t)T<<R);
    int d = 0, i = ti;
    while (d < K && d <= qi) {
        int c = q[bi*T+qi-d];
        if (flipi != -1 && qi-d == flipi) c ^= 1<<flipc;
        int ni = i;
        for (int r = 0; r < CR; r++) {
            ni = nxt[(d*CR+r)*(M<<R)+(size_t)(c>>r*R&((1<<R)-1))*T+ni];
            if (ni == -1) goto finish;
        }
        if (mint != -1 && time[(size_t)(d+1)*M+bi*T+ni] < mint) break;
        i = ni;
        d++;
    }
finish:
    return {d, time[(size_t)d*M+bi*T+i]};
}

__global__ void forward_query(size_t M, uchar*q, uchar*v, NxtInt*nxt, int*time, uchar*y) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < M) {
        int bi = t/T, ti = t%T;
        y[t] = v[(size_t)bi*T+query_trie(M, bi, ti, ti, q, nxt, time).y];
    }
}

__global__ void forward_query_with_match_len(size_t M, uchar*q, uchar*v, NxtInt*nxt, int*time, uchar*y, int*match_len) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < M) {
        int bi = t/T, ti = t%T;
        int2 match = query_trie(M, bi, ti, ti, q, nxt, time);
        y[t] = v[(size_t)bi*T+match.y];
        match_len[t] = match.x;
    }
}

template<typename T> void swap(T&a, T&b) { T tmp = a; a = b; b = tmp; }

template<int S>
struct StrideIt {
    __device__ uint64_t operator()(uint64_t i) { return i*S; }
};

template<int S>
struct RepeatKey1 {
    __device__ uint64_t operator()(uint64_t i) { return i/S; }
};

template<int S, int R_>
struct RepeatKey2 {
    KeyInt* key;
    __device__ KeyInt operator()(uint64_t i) { return key[i/((size_t)S<<R_)*S + i%S]; }
};

template <typename T>
auto make_trans(T f) { return thrust::make_transform_iterator(thrust::counting_iterator<uint64_t>{0}, f); }
template <int S, int R>
auto make_repeat_key(KeyInt*key) { return thrust::make_zip_iterator(thrust::make_tuple(make_trans(RepeatKey1<S>{}), make_trans(RepeatKey2<S,R>{key}))); }

void build_ktrie(int BH, NxtInt*nxt, int*time, uchar*k, uchar*buf) {
    static_assert(T%2 == 0); // Implicitly assumed for pointer alignment some places
    size_t M = (size_t)BH*T;
    int threads = 256, blocks = (M-1)/threads+1;

    KeyInt *key = (KeyInt*)buf; buf += M*sizeof(KeyInt);
    KeyInt *nkey = (KeyInt*)buf; buf += M*sizeof(KeyInt);
    KeyInt *nkey_sorted = (KeyInt*)buf; buf += M*sizeof(KeyInt);
    int *itime = (int*)buf; buf += M*sizeof(int);
    int *tmp_time = (int*)buf; buf += M*sizeof(int);

    atoi_mod<T><<<blocks, threads>>>(M, time);
    cudaMemset(nxt, -1, K*CR*(1<<R)*M * sizeof(NxtInt));
    cudaMemset(key, 0, M*2 * sizeof(KeyInt));

    void *tmp = (void*)buf;
    size_t tmp_sz1, tmp_sz2;
    auto offsets = make_trans(StrideIt<T>{});
    cub::DeviceSegmentedRadixSort::SortPairs(nullptr, tmp_sz1, nkey, nkey_sorted, time, time + M, M, BH, offsets, offsets+1, 0, K*C);
    cub::DeviceScan::InclusiveScanByKey(nullptr, tmp_sz2, make_repeat_key<T,C>(key), nxt, nxt, thrust::maximum<NxtInt>(), M<<R);

    for (int d = 0; d < K; d++) {
        for (int r = 0; r < CR; r++) {
            NxtInt *nxt_d = nxt + (d*CR+r)*(1<<R)*M;
            // Even C/R: time[d] -> tmp_time -> time[d+1]
            // Odd C/R: time[d] -> time[d+1] -> tmp_time -> time[d+1]
            int *time1 = !r ? time + d*M : ((CR-r)%2 ? tmp_time : time + (d+1)*M);
            int *time2 = ((CR-r)%2 ? time + (d+1)*M : tmp_time);
            extend_key<T,0><<<blocks, threads>>>(M, d, r, k, time1, key, nkey);
            invert_perm<T><<<blocks, threads>>>(M, time1, itime);

            cub::DeviceSegmentedRadixSort::SortPairs(tmp, tmp_sz1, nkey, nkey_sorted, time1, time2, M, BH, offsets, offsets+1, 0, (d*CR+r+1)*R);

            scatter_nxt<T,0><<<blocks, threads>>>(M, d, nkey_sorted, time2, itime, nxt_d);

            cub::DeviceScan::InclusiveScanByKey(tmp, tmp_sz2, make_repeat_key<T,R>(key), nxt_d, nxt_d, thrust::maximum<NxtInt>(), M<<R);

            swap(key, nkey_sorted);
        }
    }
}

void cuda_forward(int B, int H, uchar*q, uchar*k, uchar*v, uchar*y, uchar*scratch) {
    static_assert(C <= 8); // q,k,v are 8 bit types
    static_assert(C%R == 0); // Radix must divide number of bits per channel
    size_t M = (size_t)B*H*T;
    NxtInt *nxt = (NxtInt*)scratch; scratch += (K*CR<<R)*M * sizeof(NxtInt);
    int *time = (int*)scratch; scratch += (K+1)*M * sizeof(int);

    build_ktrie(B*H, nxt, time, k, (uchar*)scratch);

    int threads = 256, blocks = (M-1)/threads+1;
    forward_query<<<blocks, threads>>>(M, q, v, nxt, time, y);
}

void cuda_forward_with_match_len(int B, int H, uchar*q, uchar*k, uchar*v, uchar*y, int*match_len, uchar*scratch) {
    static_assert(C <= 8); // q,k,v are 8 bit types
    static_assert(C%R == 0); // Radix must divide number of bits per channel
    size_t M = (size_t)B*H*T;
    NxtInt *nxt = (NxtInt*)scratch; scratch += (K*CR<<R)*M * sizeof(NxtInt);
    int *time = (int*)scratch; scratch += (K+1)*M * sizeof(int);

    build_ktrie(B*H, nxt, time, k, (uchar*)scratch);

    int threads = 256, blocks = (M-1)/threads+1;
    forward_query_with_match_len<<<blocks, threads>>>(M, q, v, nxt, time, y, match_len);
}


struct Coef { float data[2][C+1]; }; // __align__(16)  is slower
__device__ Coef operator+(const Coef&a, const Coef&b) {
    Coef r;
    for (int x : {0,1})
        for (int c = 0; c < C+1; c++)
            r.data[x][c] = a.data[x][c] + b.data[x][c];
    return r;
}

__global__ void query_dv_dq(size_t M, uchar*q, uchar*v, float*dy, NxtInt*nxt, int*time, int2*match_len_t, float*dv, float*dq) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= M) return;

    size_t bi = t/T, ti = t%T;
    int2 match = query_trie(M, bi, ti, ti, q, nxt, time);
    match_len_t[t] = match;
    for (int c = 0; c < C; c++)
        atomicAdd(&dv[bi*T*C+match.y*C+c], dy[t*C+c]);

    for (int qi = ti; qi < T && qi < ti+K; qi++) {
        int omatcht = query_trie(M, bi, qi, qi, q, nxt, time).y;
        for (int c = 0; c < C; c++) {
            int nmatcht = query_trie(M, bi, qi, qi, q, nxt, time, ti, c).y;
            float sum = 0, sign = q[t]>>c&1 ? -1 : 1;
            for (int vc = 0; vc < C; vc++)
                sum += sign * (float(v[bi*T+nmatcht]>>vc&1) - float(v[bi*T+omatcht]>>vc&1)) * dy[bi*T*C+qi*C+vc];
            dq[t*C+c] += sum;
        }
    }
}

__global__ void calc_q_time(size_t M, int2*match_len_t, int*time) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    int bi = i/T2, type = i/T%2, ti = i%T, t = max(0, ti-K);
    if (!type) t = min(t, match_len_t[(size_t)bi*T+ti].y);
    time[i] = T-1-t;
}

__global__ void init_acc(size_t M, int d, uchar*v, float*dy, int2*match_len_t, int*perm, Coef*acc) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    size_t bi = i/T2;
    int ti2 = perm[i], type = ti2/T, ti = ti2%T;
    int2 match_len_t_ = match_len_t[bi*T+ti];
    int len = match_len_t_.x, matcht = match_len_t_.y;
    for (int x : {0,1}) {
        if ((type == 0 && d == len+x) || (type == 1 && d >= len+x)) {
            float sign = type*2-1, sum = 0;
            uchar vt = v[bi*T+matcht];
            for (int c = 0; c < C; c++) {
                float dyc = sign*dy[(bi*T+ti)*C+c];
                acc[i].data[x][c] = dyc;
                sum -= dyc*(vt>>c&1);
            }
            acc[i].data[x][C] = sum;
        }
    }
}

__global__ void distant_promotions_dk(size_t M, const uchar* __restrict__ k, const uchar* __restrict__ v, const NxtInt* __restrict__ nxt, Coef* __restrict__ acc, const int* __restrict__ iperm0, float* __restrict__ dk) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M*C) return;

    int flipt = tid/C%T, kc = tid%C;
    if (flipt == 0 || flipt >= T-K) return;
    size_t bi = tid/(T*C);
    nxt += bi*((size_t)T2<<R);
    acc += bi*T2;

    static_assert(K <= 32); // uint32_t mask
    uint32_t mask_list[K] = {(1ll<<K)-1};
    int trie_ind_list[K] = {iperm0[bi*T2+T+flipt+K]};
    uint8_t par[K] = {0};
    int sz = 1;

    float dkt = 0;
    for (int d = 0; d <= K; d++) {
        for (int r = 0; r < CR; r++) {
            int nsz = 0;
            uint32_t nmask_list[K];
            int ntrie_ind_list[K];
            uint8_t npar[K];
            for (int j = 0; j < sz; j++) {
                uint32_t mask = mask_list[j];
                int trie_ind = trie_ind_list[j];

                int mai = flipt + (31 - __clz(mask));

                if (r == 0) {
                    int nv = v[bi*T+mai];
                    const auto&coef = acc[d*M*2+trie_ind].data;
                    for (int vc = 0; vc < C; vc++)
                        dkt += coef[0][vc] * (nv>>vc&1);
                    dkt += coef[0][C];

                    if (d) {
                        nv = v[bi*T+flipt+par[j]];
                        for (int vc = 0; vc < C; vc++)
                            dkt -= coef[1][vc] * (nv>>vc&1);
                        dkt -= coef[1][C];
                    }
                }

                if (d == K) continue;

                uint32_t cmask = 0;
                for (uint32_t p = mask; p; p &= p-1) {
                    int bit = __ffs(p)-1, idx = flipt+bit;
                    if (idx > d) cmask |= 1<<((k[bi*T+idx-d-1]^((bit==d)<<kc))>>r&((1<<R)-1));
                }

                for (uint32_t p = cmask; p; p &= p-1) {
                    int c = __ffs(p)-1;
                    int ni = nxt[(d*CR+r)*((size_t)M*2<<R)+c*T2+trie_ind];
                    if (ni == -1) continue;

                    uint32_t nmask = 0;
                    for (uint32_t p = mask; p; p &= p-1) {
                        int bit = __ffs(p)-1, idx = flipt+bit;
                        int chr = (idx > d ? ((k[bi*T+idx-d-1]^((bit==d)<<kc))>>r&((1<<R)-1))+1 : 0);
                        if (chr == c+1) nmask |= 1<<bit;
                    }

                    nmask_list[nsz] = nmask;
                    ntrie_ind_list[nsz] = ni;
                    npar[nsz] = (r ? par[j] : mai-flipt);
                    nsz++;
                }
            }
            sz = nsz;
            for (int j = 0; j < nsz; j++) {
                mask_list[j] = nmask_list[j];
                trie_ind_list[j] = ntrie_ind_list[j];
                par[j] = npar[j];
            }
        }
    }
    float sign = (k[bi*T+flipt-1]>>kc&1 ? -1 : 1);
    dk[bi*T*C+(flipt-1)*C+kc] += sign*dkt;
}

__device__ int2 max(const int2& a, const int2& b) {
    return (a.x > b.x || (a.x == b.x && a.y > b.y)) ? a : b;
}

__device__ int lcp(int ki, int qi, uchar*k, uchar*q) {
    int r = 0;
    while (ki-r > 0 && qi-r >= 0 && r < K && q[qi-r] == k[ki-r]) r++;
    return r;
};

__device__ void window_query(int y, int L, int R, uchar*k, uchar*q, int2*ret) {
    if (L > R) return;
    int LCP[K];
    int2 best = {-1, -1};
    int nx = R;
    for (int i = min(R+K-1, y); i >= L; i--) {
        int l = lcp(i, y, k, q);
        if (i <= R) LCP[i-L] = l;
        if (l > best.x) best = {l, i};
        while (nx >= max(i-1-best.x, L)) {
            ret[(nx-L)*C] = max(ret[(nx-L)*C], int2{min(best.x, best.y-nx), best.y});
            nx--;
        }
    }

    best = {-1, -1};
    for (int i = L; i <= R; i++) {
        ret[(i-L)*C] = max(ret[(i-L)*C], best);
        best = max(best, {LCP[i-L], i});
    }
    for (int i = 0; i < K; i++)
        for (int c = 1; c < C; c++)
            ret[i*C+c] = ret[i*C];

    for (int i = min(R+K-1, y); i >= L; i--) {
        int l = lcp(i, y, k, q), x = i-l;
        if (x >= L && x <= R && x > 0 && y-l >= 0) {
            int diff = k[x]^q[y-l];
            if (__popc(diff) == 1) {
                int c = __ffs(diff)-1;
                int l2 = min(l+1+lcp(x-1, y-l-1, k, q), K);
                ret[(x-L)*C+c] = max(ret[(x-L)*C+c], int2{l2, i});
            }
        }
    }
}

__global__ void local_dk(size_t M, uchar*q, uchar*k, uchar*v, float*dy, NxtInt*nxt, int*time, int2*match_len_t, float*dk) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    size_t bi = i/T;
    int y = i%T;

    uchar old_v = v[bi*T+match_len_t[bi*T+y].y];
    int2 crop_match = (y >= K ? query_trie(M, bi, y-K, y, q, nxt, time) : int2{0,0});

    int L = max(y-K+1,1);
    int2 local[C*K] = {};
    window_query(y, L, y, k+bi*T-1, q+bi*T, local);

    for (int x = L; x <= y; x++) {
        for (int kc = 0; kc < C; kc++) {
            int nmatch_t = max(local[(x-L)*C+kc], crop_match).y;
            float add = 0;
            for (int vc = 0; vc < C; vc++)
                add += dy[bi*T*C+y*C+vc] * ((float)(v[bi*T+nmatch_t]>>vc&1) - (old_v>>vc&1));
            float sign = (k[bi*T+x-1]>>kc&1 ? -1 : 1);
            atomicAdd(&dk[bi*T*C+(x-1)*C+kc], add*sign);
        }
    }
}

__global__ void distant_demotions_dk(size_t M, uchar*q, uchar*k, uchar*v, float*dy, NxtInt*nxt, int*time, int2*match_len_t, float*dk) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    size_t bi = i/T;
    int y = i%T;

    auto [old_len, old_t] = match_len_t[bi*T+y];
    uchar old_v = v[bi*T+old_t];
    int2 crop_match = (old_t >= K ? query_trie(M, bi, old_t-K, y, q, nxt, time) : int2{0,0});
    int2 after_match = query_trie(M, bi, y, y, q, nxt, time, -1, -1, min(old_t+K,y));
    int2 hole_match = max(crop_match, after_match);

    int L = max(old_t-K+1, 1), R = min(old_t, y-K);
    int2 window[C*K] = {};
    window_query(y, L, R, k+bi*T-1, q+bi*T, window);

    for (int x = L; x <= R; x++) {
        for (int kc = 0; kc < C; kc++) {
            auto [new_len, new_t] = max(window[(x-L)*C+kc], hole_match);
            if (old_len >= new_len) {
                float add = 0;
                for (int vc = 0; vc < C; vc++)
                    add += dy[bi*T*C+y*C+vc] * ((float)(v[bi*T+new_t]>>vc&1) - (old_v>>vc&1));
                float sign = (k[bi*T+x-1]>>kc&1 ? -1 : 1);
                atomicAdd(&dk[bi*T*C+(x-1)*C+kc], add*sign);
            }
        }
    }
}

void build_qtrie(int BH, uchar*q, uchar*v, float*dy, int2*match_len_t, NxtInt*nxt, Coef*acc, int*iperm0, int*buf) {
    size_t M = (size_t)BH*T2;
    int threads = 256, blocks = (M-1)/threads+1;

    int *perm = &buf[M*0];
    int *time = &buf[M*1];
    int *nperm = &buf[M*2];
    int *ntime = &buf[M*3];
    int *iperm = &buf[M*4];
    KeyInt*buf2 = (KeyInt*)&buf[M*5];
    KeyInt *key = &buf2[M*0];
    KeyInt *nkey = &buf2[M*1];
    KeyInt *nkey_sorted = &buf2[M*2];

    atoi_mod<T2><<<blocks, threads>>>(M, perm);
    calc_q_time<<<blocks, threads>>>(M, match_len_t, time);
    cudaMemset(nxt, -1, ((K+1)*CR<<R)*M * sizeof(NxtInt));
    cudaMemset(acc, 0, (K+1)*M * sizeof(Coef));
    cudaMemset(key, 0, M * sizeof(KeyInt));

    void *tmp = &buf2[M*3];
    size_t tmp_sz0, tmp_sz1, tmp_sz2, tmp_sz3;
    auto offsets = make_trans(StrideIt<T2>{});
    cub::DeviceSegmentedRadixSort::SortPairs(nullptr, tmp_sz0, time, ntime, perm, nperm, M, BH, offsets, offsets+1);
    cub::DeviceSegmentedRadixSort::SortPairs(nullptr, tmp_sz1, nkey, nkey_sorted, perm, nperm, M, BH, offsets, offsets+1);
    cub::DeviceScan::InclusiveScanByKey(nullptr, tmp_sz2, make_repeat_key<T2,R>(key), nxt, nxt, thrust::maximum<NxtInt>(), M<<R);
    cub::DeviceScan::InclusiveScanByKey(nullptr, tmp_sz3, make_repeat_key<T2,0>(nkey_sorted), acc, acc, thrust::plus<Coef>(), M);
    assert(max(max(max(tmp_sz0, tmp_sz1), tmp_sz2), tmp_sz3) == max(max(tmp_sz1, tmp_sz2), tmp_sz3));

    cub::DeviceSegmentedRadixSort::SortPairs(tmp, tmp_sz0, time, ntime, perm, nperm, M, BH, offsets, offsets+1);
    swap(perm, nperm);

    for (int d = 0; d < K+1; d++) {
        Coef *acc_d = acc + d*M;
        init_acc<<<blocks, threads>>>(M, d, v, dy, match_len_t, perm, acc_d);
        cub::DeviceScan::InclusiveScanByKey(tmp, tmp_sz3, make_repeat_key<T2,0>(key), acc_d, acc_d, thrust::plus<Coef>(), M);
        for (int r = 0; r < CR; r++) {
            NxtInt *nxt_d = nxt + (d*CR+r)*(1<<R)*M;
            extend_key<T2,1><<<blocks, threads>>>(M, d, r, q, perm, key, nkey);
            cub::DeviceSegmentedRadixSort::SortPairs(tmp, tmp_sz1, nkey, nkey_sorted, perm, nperm, M, BH, offsets, offsets+1, 0, (d*CR+r+1)*R);

            invert_perm<T2><<<blocks, threads>>>(M, perm, iperm);
            if (!d && !r) cudaMemcpy(iperm0, iperm, M * sizeof(int), cudaMemcpyDeviceToDevice);
            scatter_nxt<T2,1><<<blocks, threads>>>(M, d, nkey_sorted, nperm, iperm, nxt_d);

            cub::DeviceScan::InclusiveScanByKey(tmp, tmp_sz2, make_repeat_key<T2,R>(key), nxt_d, nxt_d, thrust::maximum<NxtInt>(), M<<R);

            swap(perm, nperm);
            swap(key, nkey_sorted);
        }
    }
}

int64_t scratch_size(int64_t B, int64_t H) {
    size_t M = (size_t)B*H*T, mem, mem2, mem3;
    auto offsets = make_trans(StrideIt<T2>{});
    cub::DeviceSegmentedRadixSort::SortPairs(nullptr, mem, (KeyInt*)nullptr, (KeyInt*)nullptr, (int*)nullptr, (int*)nullptr, M*2, B*H, offsets, offsets+1); // cub tmp storage
    cub::DeviceScan::InclusiveScanByKey(nullptr, mem2, make_repeat_key<T2,R>(nullptr), (int*)nullptr, (int*)nullptr, thrust::maximum<NxtInt>(), M*2<<R);
    cub::DeviceScan::InclusiveScanByKey(nullptr, mem3, make_repeat_key<T2,0>(nullptr), (Coef*)nullptr, (Coef*)nullptr, thrust::plus<Coef>(), M*2);
    mem = max(max(mem,mem2),mem3) + ((K+1)*CR<<R)*M*2*sizeof(NxtInt) + (1+1+5)*M*2*sizeof(int) + 3*M*2*sizeof(KeyInt) + (K+1)*M*2*sizeof(Coef); // nxt, match_len_t, iperm0, build_qtrie's buf, acc
    return mem;
}

void cuda_backward(int B, int H, uchar*q, uchar*k, uchar*v, float*dy, float*dq, float*dk, float*dv, uchar*scratch) {
    size_t M = (size_t)B*H*T;

    int2 *match_len_t = (int2*)scratch; scratch += M*sizeof(int2);

    uchar *buf = (uchar*)scratch;

    NxtInt *nxt = (NxtInt*)buf; buf += (K*CR<<R)*M * sizeof(NxtInt);
    int *time = (int*)buf; buf += (K+1)*M * sizeof(int);

    cudaMemset(dq, 0, M*C * sizeof(float));
    cudaMemset(dk, 0, M*C * sizeof(float));
    cudaMemset(dv, 0, M*C * sizeof(float));

    build_ktrie(B*H, nxt, time, k, buf);

    int threads = 256, blocks = (M-1)/threads+1;
    query_dv_dq<<<blocks, threads>>>(M, q, v, dy, nxt, time, match_len_t, dv, dq);
    local_dk<<<blocks, threads>>>(M, q, k, v, dy, nxt, time, match_len_t, dk);
    distant_demotions_dk<<<blocks, threads>>>(M, q, k, v, dy, nxt, time, match_len_t, dk);

    buf = (uchar*)scratch;
    nxt = (NxtInt*)buf; buf += ((K+1)*CR<<R)*M*2 * sizeof(NxtInt);
    int *iperm0 = (int*)buf; buf += M*2 * sizeof(int);;
    Coef *acc = (Coef*)buf; buf += (K+1)*M*2 * sizeof(Coef);

    build_qtrie(B*H, q, v, dy, match_len_t, nxt, acc, iperm0, (int*)buf);

    distant_promotions_dk<<<(M*C-1)/threads+1, threads>>>(M, k, v, nxt, acc, iperm0, dk);
}

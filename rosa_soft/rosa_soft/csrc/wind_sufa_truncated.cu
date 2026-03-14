#include "../../../wind_rosa/wind_rosa.cu"


__global__ void forward_query_with_metadata(
    size_t M,
    uchar* q,
    uchar* v,
    NxtInt* nxt,
    int* time,
    uchar* y,
    int* match_len,
    int* endpos
) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < M) {
        int bi = t / T, ti = t % T;
        int2 match = query_trie(M, bi, ti, ti, q, nxt, time);
        y[t] = v[(size_t)bi * T + match.y];
        match_len[t] = match.x;
        endpos[t] = match.x > 0 ? (match.y - 1) : -1;
    }
}


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
) {
    static_assert(C <= 8);
    static_assert(C % R == 0);
    size_t M = (size_t)B * H * T;
    NxtInt* nxt = (NxtInt*)scratch;
    scratch += (K * CR << R) * M * sizeof(NxtInt);
    int* time = (int*)scratch;
    scratch += (K + 1) * M * sizeof(int);

    build_ktrie(B * H, nxt, time, k, (uchar*)scratch);

    int threads = 256, blocks = (M - 1) / threads + 1;
    forward_query_with_metadata<<<blocks, threads>>>(M, q, v, nxt, time, y, match_len, endpos);
}

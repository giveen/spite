/*
 * kernels/qwen/qwen3_8/nvidia/sm_60/kvflash_pascal.cuh
 *
 * Pascal (GP100, sm_60) flash-decode tile for the Qwen3.8 attention op.
 * Plugs into kv_attn_flash.inl through the SPITE_KVFLASH_ARCH hook, so the
 * prologue (QK-norm, RoPE, KV-row write), chunking, split-K workspace, combine
 * pass and VBR fallback are the shared vendor code, unchanged.
 *
 * Same algorithm and numerics as kvflash_core_portable (float staging, same
 * FMA order); the difference is shared-memory footprint.  The portable tile
 * keeps a K tile and a V tile resident at once:
 *
 *   hd=128: s_q 2 KB + s_k 16.1 KB + s_v 16.1 KB + s_p 0.5 KB = 34.75 KB
 *
 * GP100 has 64 KB of shared memory per SM, so that is ONE resident block
 * (4 warps) per SM.  K is dead once the scores are computed, so this tile
 * stages V into the same buffer after one extra barrier:
 *
 *   hd=128: s_q 2 KB + s_kv 16.1 KB + s_p 0.5 KB = 18.6 KB  -> 3 blocks / SM
 *   hd=256: s_q 4 KB + s_kv 32.1 KB + s_p 0.5 KB = 36.6 KB  -> fits Pascal's
 *           48 KB static per-block limit (the two-buffer tile would need 68 KB)
 *
 * Included after kv_attn.inl and before kv_attn_flash.inl, with
 * SPITE_KVFLASH_ARCH, KVFLASH_TILE and KVFLASH_WARPS defined by the includer.
 */

#pragma once

#ifndef SPITE_KVFLASH_ARCH
#error "define SPITE_KVFLASH_ARCH before including kvflash_pascal.cuh"
#endif

/* Decode KVFLASH_TILE rows of one KV head (kh) into s[t][0..HD) as float.
 * Warp w owns rows w, w+nwarps, ...; lanes walk a row (coalesced global side,
 * conflict-free smem side thanks to the +1 padding). */
template <int HD>
__device__ __forceinline__ void kvp_stage(float (*s)[HD + 1], const uint8_t* __restrict__ base,
                                          int row_bytes, int kind, int kh, int t0, int t_end,
                                          int w, int nwarps, int lane) {
    for (int t = w; t < KVFLASH_TILE; t += nwarps) {
        float* row = s[t];
        if (t0 + t >= t_end) {
#pragma unroll
            for (int k = 0; k < HD / 32; ++k) row[lane + 32 * k] = 0.0f;
            continue;
        }
        const uint8_t* r = base + static_cast<size_t>(t0 + t) * row_bytes;
        if (kind == SPITE_TYPE_F32) {
            const float* f = reinterpret_cast<const float*>(r) + static_cast<size_t>(kh) * HD;
#pragma unroll
            for (int k = 0; k < HD / 32; ++k) row[lane + 32 * k] = f[lane + 32 * k];
        } else if (kind == SPITE_TYPE_F16) {
            const __half* f = reinterpret_cast<const __half*>(r) + static_cast<size_t>(kh) * HD;
#pragma unroll
            for (int k = 0; k < HD / 32; ++k) row[lane + 32 * k] = __half2float(f[lane + 32 * k]);
        } else if (kind == SPITE_TYPE_Q8_0) {
            const BlockQ8_0* b = reinterpret_cast<const BlockQ8_0*>(r) + static_cast<size_t>(kh) * (HD / 32);
#pragma unroll
            for (int k = 0; k < HD / 32; ++k)
                row[lane + 32 * k] = __half2float(b[k].d) * static_cast<float>(b[k].qs[lane]);
        } else if (kind == SPITE_TYPE_Q5_1) {
            const KvqQ5_1* b = reinterpret_cast<const KvqQ5_1*>(r) + static_cast<size_t>(kh) * (HD / 32);
            const int j = lane & 15;
#pragma unroll
            for (int k = 0; k < HD / 32; ++k) {
                const uint8_t q = b[k].qs[j];
                const uint32_t lo = (lane < 16) ? (q & 0x0Fu) : (q >> 4);
                const uint32_t v = lo | (((b[k].qh >> lane) & 1u) << 4);
                row[lane + 32 * k] =
                    static_cast<float>(v) * __half2float(b[k].d) + __half2float(b[k].m);
            }
        } else { /* SPITE_TYPE_Q4_0 */
            const KvqQ4_0* b = reinterpret_cast<const KvqQ4_0*>(r) + static_cast<size_t>(kh) * (HD / 32);
            const int j = lane & 15;
#pragma unroll
            for (int k = 0; k < HD / 32; ++k) {
                const uint8_t q = b[k].qs[j];
                const int lo = (lane < 16) ? (q & 0x0F) : (q >> 4);
                row[lane + 32 * k] = static_cast<float>(lo - 8) * __half2float(b[k].d);
            }
        }
    }
}

template <int HD>
__global__ void kvflash_core_pascal(const float* __restrict__ q, const uint8_t* __restrict__ kc,
                                    const uint8_t* __restrict__ vc, int row_bytes, int kind,
                                    int nh, int group, int n_tok, int per, float scale,
                                    float* __restrict__ wm, float* __restrict__ wl,
                                    float* __restrict__ wacc) {
    static_assert(HD % 32 == 0, "pascal flash tile: head_dim must be a multiple of 32");
    constexpr int NLANE = HD / 32;
    const int lane = threadIdx.x;
    const int w = threadIdx.y;
    const int nwarps = blockDim.y;
    const int gchunks = (group + KVFLASH_WARPS - 1) / KVFLASH_WARPS;

    __shared__ float s_q[KVFLASH_WARPS][HD];
    __shared__ float s_kv[KVFLASH_TILE][HD + 1];  /* K during scoring, then V */
    __shared__ float s_p[KVFLASH_WARPS][KVFLASH_TILE];

    const int kh = blockIdx.y / gchunks;
    const int hw = blockIdx.y % gchunks * nwarps + w;
    const bool live = hw < group;
    const int h = kh * group + hw;

    const int t_begin = blockIdx.x * per;
    const int t_end = min(n_tok, t_begin + per);

    if (live)
        for (int i = lane; i < HD; i += 32) s_q[w][i] = q[static_cast<size_t>(h) * HD + i];
    __syncwarp();

    float m = -INFINITY, l = 0.0f;
    float acc[NLANE];
#pragma unroll
    for (int e = 0; e < NLANE; ++e) acc[e] = 0.0f;

    for (int t0 = t_begin; t0 < t_end; t0 += KVFLASH_TILE) {
        kvp_stage<HD>(s_kv, kc, row_bytes, kind, kh, t0, t_end, w, nwarps, lane);
        __syncthreads();

        const int gt = t0 + lane;
        float s = -INFINITY;
        if (live && gt < t_end) {
            const float* krow = s_kv[lane];
            const float* qrow = s_q[w];
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
            for (int i = 0; i < HD; i += 4) {
                a0 = fmaf(qrow[i + 0], krow[i + 0], a0);
                a1 = fmaf(qrow[i + 1], krow[i + 1], a1);
                a2 = fmaf(qrow[i + 2], krow[i + 2], a2);
                a3 = fmaf(qrow[i + 3], krow[i + 3], a3);
            }
            s = (a0 + a1 + a2 + a3) * scale;
        }

        const float m_new = fmaxf(m, warp_max(s));
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + warp_sum(p);
        m = m_new;
#pragma unroll
        for (int e = 0; e < NLANE; ++e) acc[e] *= alpha;
        if (live) s_p[w][lane] = p;
        __syncthreads();  /* every warp is done reading K before V overwrites it */

        kvp_stage<HD>(s_kv, vc, row_bytes, kind, kh, t0, t_end, w, nwarps, lane);
        __syncthreads();

        if (live) {
            for (int t = 0; t < KVFLASH_TILE; ++t) {
                if (t0 + t >= t_end) break;
                const float pv = s_p[w][t];
                const float* vrow = s_kv[t];
#pragma unroll
                for (int e = 0; e < NLANE; ++e) acc[e] = fmaf(pv, vrow[lane + 32 * e], acc[e]);
            }
        }
        __syncthreads();
    }

    if (live) {
        if (lane == 0) {
            wm[static_cast<size_t>(blockIdx.x) * nh + h] = m;
            wl[static_cast<size_t>(blockIdx.x) * nh + h] = l;
        }
#pragma unroll
        for (int e = 0; e < NLANE; ++e)
            wacc[(static_cast<size_t>(blockIdx.x) * nh + h) * HD + lane + 32 * e] = acc[e];
    }
}

/* hd=256 fits only because of the shared K/V buffer; anything else falls back
 * to the VBR back end inside kvflash_run(). */
inline bool kvflash_hd_supported(int hd) {
    return hd == 32 || hd == 64 || hd == 96 || hd == 128 || hd == 256;
}

template <int HD>
inline int kvflash_launch_pascal(const KvattnArgs& a, int chunks, int per, float* acc, float* m,
                                 float* l, cudaStream_t s) {
    const int gchunks = (a.group + KVFLASH_WARPS - 1) / KVFLASH_WARPS;
    const dim3 grid(chunks, a.nkv * gchunks);
    const dim3 block(32, KVFLASH_WARPS);
    kvflash_core_pascal<HD><<<grid, block, 0, s>>>(a.q, a.kc, a.vc, a.row_bytes, a.kind, a.nh,
                                                   a.group, a.n_tok, per, a.scale, m, l, acc);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

inline int kvflash_launch_hd(const KvattnArgs& a, int chunks, int per, float* acc, float* m,
                             float* l, cudaStream_t s) {
    switch (a.hd) {
    case 32: return kvflash_launch_pascal<32>(a, chunks, per, acc, m, l, s);
    case 64: return kvflash_launch_pascal<64>(a, chunks, per, acc, m, l, s);
    case 96: return kvflash_launch_pascal<96>(a, chunks, per, acc, m, l, s);
    case 128: return kvflash_launch_pascal<128>(a, chunks, per, acc, m, l, s);
    case 256: return kvflash_launch_pascal<256>(a, chunks, per, acc, m, l, s);
    default: return -1;
    }
}

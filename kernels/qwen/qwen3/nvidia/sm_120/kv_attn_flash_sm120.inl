/*
 * kernels/qwen/qwen3/nvidia/sm_120/kv_attn_flash_sm120.inl
 *
 * Blackwell (sm_120) tile kernel for the flash-decoding attention back end.
 * Scope: all sm_120-family GPUs.  The algorithm (split-KV flash decoding, GQA
 * group tile reuse, fused online softmax) is not what changes here — the
 * vendor root in ../kv_attn_flash.inl owns that and every NVIDIA card gets it.
 * What changes is how the tile kernel uses the hardware:
 *
 *   • 128-bit transactions end to end: the KV tile is staged with float4 loads
 *     (F32) or __half2 loads (F16), the Q.K dot product reads a float4 per
 *     lane, and the P.V accumulation owns a whole float4 of output dims, so
 *     the hot loops issue a quarter of the shared-memory instructions.
 *   • Row stride padded by 4 floats instead of 1: float4 needs 16-byte aligned
 *     rows, and a stride that is a multiple of 4 words still keeps every
 *     scalar access in the kernel on consecutive banks.
 *   • The KV tier is compiled in, not branched on: each block-tier decoder
 *     (q8_0/q5_1/q4_0) is specialized, which removes kvq_get's per-element
 *     tier switch and its per-element block index math from the staging loop —
 *     the loop index *is* the block index, because a lane walks exactly one
 *     element per 32 positions.
 *
 * The metric that motivated it: with the portable tile kernel, staging a
 * quantized tier costs more than the attention maths itself (q8_0 measured
 * ~2x f32 at n_tok=8192 on RTX 5090), because kvq_get is a runtime switch
 * evaluated once per staged element.
 *
 * Included from sm_120/kernel.cu *before* ../kv_attn_flash.inl, with
 * SPITE_KVFLASH_ARCH defined: this file supplies the tile kernel and the
 * hd dispatch, the shared file supplies the chunking, the split-K workspace,
 * the combine pass and kvflash_run().  The decoders mirror kvq_get() in
 * ../kv_attn.inl element for element, and that agreement is what the verify
 * tool checks against the generic C reference.
 *
 * Uses (from the including kernel.cu, like kv_attn.inl): warp_max, warp_sum,
 * threads_for, stream_of, launch_matvec.
 */

#ifndef SPITE_QWEN3_KV_ATTN_FLASH_SM120_INL
#define SPITE_QWEN3_KV_ATTN_FLASH_SM120_INL

/* Tile geometry this level owns; the shared scaffolding reads these macros for
 * its chunking heuristic, so they must be defined before it is included.  A
 * warp-wide score step fixes the tile at one KV row per lane. */
#ifndef KVFLASH_TILE
#define KVFLASH_TILE 32
#endif
#ifndef KVFLASH_WARPS
#define KVFLASH_WARPS 4
#endif

template <int HD>
__global__ void kvflash_core_sm120(const float* __restrict__ q, const uint8_t* __restrict__ kc,
                                   const uint8_t* __restrict__ vc, int row_bytes, int kind,
                                   int nh, int group, int n_tok, int per, float scale,
                                   float* __restrict__ wm, float* __restrict__ wl,
                                   float* __restrict__ wacc) {
    /* The block decoders below index a head's VBR blocks from the head offset,
     * which only holds when a head range starts on a 32-element block boundary;
     * the same multiple of 32 makes the lane->dim mapping exact. */
    static_assert(HD % 32 == 0, "flash tile: head_dim must be a multiple of 32");
    /* A lane owns 4 consecutive output dims once it can fill a float4, and
     * strided single dims otherwise. */
    constexpr bool VEC4 = (HD % 128) == 0;
    constexpr int ND = VEC4 ? HD / 128 : HD / 32;

    const int lane = threadIdx.x;
    const int w = threadIdx.y;
    const int nwarps = blockDim.y;
    const int gchunks = (group + KVFLASH_WARPS - 1) / KVFLASH_WARPS;

    __shared__ float s_q[KVFLASH_WARPS][HD + 4];
    __shared__ float s_k[KVFLASH_TILE][HD + 4];
    __shared__ float s_v[KVFLASH_TILE][HD + 4];
    __shared__ float s_p[KVFLASH_WARPS][KVFLASH_TILE];

    const int kh = blockIdx.y / gchunks;
    const int hw = blockIdx.y % gchunks * nwarps + w;
    const bool live = hw < group;
    const int h = kh * group + hw;

    const int t_begin = blockIdx.x * per;
    const int t_end = min(n_tok, t_begin + per);

    if (live) {
        const float4* q4 = reinterpret_cast<const float4*>(q + static_cast<size_t>(h) * HD);
        float4* sq4 = reinterpret_cast<float4*>(s_q[w]);
#pragma unroll
        for (int k = lane; k < HD / 4; k += 32) sq4[k] = q4[k];
    }
    __syncwarp();

    float m = -INFINITY, l = 0.0f;
    float4 acc4[VEC4 ? ND : 1];
    float acc1[VEC4 ? 1 : ND];
    if constexpr (VEC4) {
#pragma unroll
        for (int e = 0; e < ND; ++e) acc4[e] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
#pragma unroll
        for (int e = 0; e < ND; ++e) acc1[e] = 0.0f;
    }

    for (int t0 = t_begin; t0 < t_end; t0 += KVFLASH_TILE) {
        /* ── stage K and V: coalesced wide reads, decoded once per group ─── */
        for (int t = w; t < KVFLASH_TILE; t += nwarps) {
            const int gt = t0 + t;
            const bool ok = gt < t_end;
            const uint8_t* kr = kc + static_cast<size_t>(gt) * row_bytes;
            const uint8_t* vr = vc + static_cast<size_t>(gt) * row_bytes;
            float* skrow = s_k[t];
            float* svrow = s_v[t];

            if (kind == SPITE_TYPE_F32) {
                const float* kf = reinterpret_cast<const float*>(kr) + static_cast<size_t>(kh) * HD;
                const float* vf = reinterpret_cast<const float*>(vr) + static_cast<size_t>(kh) * HD;
#pragma unroll
                for (int k = lane; k < HD / 4; k += 32) {
                    const float4 kv =
                        ok ? reinterpret_cast<const float4*>(kf)[k] : make_float4(0, 0, 0, 0);
                    const float4 vv =
                        ok ? reinterpret_cast<const float4*>(vf)[k] : make_float4(0, 0, 0, 0);
                    reinterpret_cast<float4*>(skrow)[k] = kv;
                    reinterpret_cast<float4*>(svrow)[k] = vv;
                }
            } else if (kind == SPITE_TYPE_F16) {
                const __half2* kh2 = reinterpret_cast<const __half2*>(
                    reinterpret_cast<const __half*>(kr) + static_cast<size_t>(kh) * HD);
                const __half2* vh2 = reinterpret_cast<const __half2*>(
                    reinterpret_cast<const __half*>(vr) + static_cast<size_t>(kh) * HD);
#pragma unroll
                for (int k = lane; k < HD / 2; k += 32) {
                    const float2 kv = ok ? __half22float2(kh2[k]) : make_float2(0.0f, 0.0f);
                    const float2 vv = ok ? __half22float2(vh2[k]) : make_float2(0.0f, 0.0f);
                    reinterpret_cast<float2*>(skrow)[k] = kv;
                    reinterpret_cast<float2*>(svrow)[k] = vv;
                }
            } else if (kind == SPITE_TYPE_Q8_0) {
                const BlockQ8_0* kb =
                    reinterpret_cast<const BlockQ8_0*>(kr) + static_cast<size_t>(kh) * (HD / 32);
                const BlockQ8_0* vb =
                    reinterpret_cast<const BlockQ8_0*>(vr) + static_cast<size_t>(kh) * (HD / 32);
#pragma unroll
                for (int k = 0; k < HD / 32; ++k) {
                    skrow[lane + 32 * k] =
                        ok ? __half2float(kb[k].d) * static_cast<float>(kb[k].qs[lane]) : 0.0f;
                    svrow[lane + 32 * k] =
                        ok ? __half2float(vb[k].d) * static_cast<float>(vb[k].qs[lane]) : 0.0f;
                }
            } else if (kind == SPITE_TYPE_Q5_1) {
                const KvqQ5_1* kb =
                    reinterpret_cast<const KvqQ5_1*>(kr) + static_cast<size_t>(kh) * (HD / 32);
                const KvqQ5_1* vb =
                    reinterpret_cast<const KvqQ5_1*>(vr) + static_cast<size_t>(kh) * (HD / 32);
#pragma unroll
                for (int k = 0; k < HD / 32; ++k) {
                    if (ok) {
                        const int j = lane & 15;
                        const uint8_t kbq = kb[k].qs[j];
                        const uint8_t vbq = vb[k].qs[j];
                        const uint32_t klo = (lane < 16) ? (kbq & 0x0Fu) : (kbq >> 4);
                        const uint32_t vlo = (lane < 16) ? (vbq & 0x0Fu) : (vbq >> 4);
                        const uint32_t kq = klo | (((kb[k].qh >> lane) & 1u) << 4);
                        const uint32_t vq = vlo | (((vb[k].qh >> lane) & 1u) << 4);
                        /* KvqQ5_1 is {half d; half m; u32 qh; u8 qs[16]}, so d and
                         * m load as one aligned half2 — 24-byte blocks stay 4-byte
                         * aligned for every k and every head offset. */
                        const float2 kdm = __half22float2(
                            *reinterpret_cast<const __half2*>(&kb[k].d));
                        const float2 vdm = __half22float2(
                            *reinterpret_cast<const __half2*>(&vb[k].d));
                        skrow[lane + 32 * k] = static_cast<float>(kq) * kdm.x + kdm.y;
                        svrow[lane + 32 * k] = static_cast<float>(vq) * vdm.x + vdm.y;
                    } else {
                        skrow[lane + 32 * k] = 0.0f;
                        svrow[lane + 32 * k] = 0.0f;
                    }
                }
            } else { /* SPITE_TYPE_Q4_0 */
                const KvqQ4_0* kb =
                    reinterpret_cast<const KvqQ4_0*>(kr) + static_cast<size_t>(kh) * (HD / 32);
                const KvqQ4_0* vb =
                    reinterpret_cast<const KvqQ4_0*>(vr) + static_cast<size_t>(kh) * (HD / 32);
#pragma unroll
                for (int k = 0; k < HD / 32; ++k) {
                    if (ok) {
                        const int j = lane & 15;
                        const uint8_t kbq = kb[k].qs[j];
                        const uint8_t vbq = vb[k].qs[j];
                        const int klo = (lane < 16) ? (kbq & 0x0F) : (kbq >> 4);
                        const int vlo = (lane < 16) ? (vbq & 0x0F) : (vbq >> 4);
                        skrow[lane + 32 * k] = static_cast<float>(klo - 8) * __half2float(kb[k].d);
                        svrow[lane + 32 * k] = static_cast<float>(vlo - 8) * __half2float(vb[k].d);
                    } else {
                        skrow[lane + 32 * k] = 0.0f;
                        svrow[lane + 32 * k] = 0.0f;
                    }
                }
            }
        }
        __syncthreads();

        /* ── scores: one float4 per lane per step, four independent chains ── */
        const int gt = t0 + lane;
        float s = -INFINITY;
        if (live && gt < t_end) {
            const float4* krow4 = reinterpret_cast<const float4*>(s_k[lane]);
            const float4* qrow4 = reinterpret_cast<const float4*>(s_q[w]);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
            for (int k = 0; k < HD / 4; ++k) {
                const float4 qv = qrow4[k];
                const float4 kv = krow4[k];
                a0 = fmaf(qv.x, kv.x, a0);
                a1 = fmaf(qv.y, kv.y, a1);
                a2 = fmaf(qv.z, kv.z, a2);
                a3 = fmaf(qv.w, kv.w, a3);
            }
            s = (a0 + a1 + a2 + a3) * scale;
        }

        /* ── online softmax (same recurrence as the portable back end) ────── */
        const float m_new = fmaxf(m, warp_max(s));
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + warp_sum(p);
        m = m_new;
        if constexpr (VEC4) {
#pragma unroll
            for (int e = 0; e < ND; ++e) {
                acc4[e].x *= alpha;
                acc4[e].y *= alpha;
                acc4[e].z *= alpha;
                acc4[e].w *= alpha;
            }
        } else {
#pragma unroll
            for (int e = 0; e < ND; ++e) acc1[e] *= alpha;
        }
        if (live) s_p[w][lane] = p;
        __syncwarp();

        /* ── P.V: a whole float4 of output dims per smem transaction ─────── */
        if (live) {
            for (int t = 0; t < KVFLASH_TILE; ++t) {
                if (t0 + t >= t_end) break;
                const float pv = s_p[w][t];
                const float* vrow = s_v[t];
                if constexpr (VEC4) {
#pragma unroll
                    for (int e = 0; e < ND; ++e) {
                        const float4 vv = reinterpret_cast<const float4*>(vrow)[lane + 32 * e];
                        acc4[e].x = fmaf(pv, vv.x, acc4[e].x);
                        acc4[e].y = fmaf(pv, vv.y, acc4[e].y);
                        acc4[e].z = fmaf(pv, vv.z, acc4[e].z);
                        acc4[e].w = fmaf(pv, vv.w, acc4[e].w);
                    }
                } else {
#pragma unroll
                    for (int e = 0; e < ND; ++e)
                        acc1[e] = fmaf(pv, vrow[lane + 32 * e], acc1[e]);
                }
            }
        }
        __syncthreads();
    }

    /* Hand the partial (m, l, acc) to the combine pass.  A chunk past the end
     * of the KV history contributes nothing: m stays -inf, so its combine
     * weight exp(m - M) is 0. */
    if (live) {
        if (lane == 0) {
            wm[static_cast<size_t>(blockIdx.x) * nh + h] = m;
            wl[static_cast<size_t>(blockIdx.x) * nh + h] = l;
        }
        float* dst = wacc + (static_cast<size_t>(blockIdx.x) * nh + h) * HD;
        if constexpr (VEC4) {
#pragma unroll
            for (int e = 0; e < ND; ++e)
                reinterpret_cast<float4*>(dst)[lane + 32 * e] = acc4[e];
        } else {
#pragma unroll
            for (int e = 0; e < ND; ++e) {
                const int d = lane + 32 * e;
                if (d < HD) dst[d] = acc1[e];
            }
        }
    }
}

/* Same coverage as the portable tile kernel: multiples of 32, where the head
 * range starts on a VBR block boundary and the lane->dim mapping is exact.
 * Other head dims take the VBR back end via kvflash_run()'s fallback. */
inline bool kvflash_hd_supported(int hd) {
    return hd == 32 || hd == 64 || hd == 96 || hd == 128;
}

template <int HD>
inline int kvflash_launch_tile(const KvattnArgs& a, int chunks, int per, float* acc, float* m,
                               float* l, cudaStream_t s) {
    const int gchunks = (a.group + KVFLASH_WARPS - 1) / KVFLASH_WARPS;
    const dim3 grid(chunks, a.nkv * gchunks);
    const dim3 block(32, KVFLASH_WARPS);
    kvflash_core_sm120<HD><<<grid, block, 0, s>>>(a.q, a.kc, a.vc, a.row_bytes, a.kind, a.nh,
                                                  a.group, a.n_tok, per, a.scale, m, l, acc);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

inline int kvflash_launch_hd(const KvattnArgs& a, int chunks, int per, float* acc, float* m,
                             float* l, cudaStream_t s) {
    switch (a.hd) {
    case 32: return kvflash_launch_tile<32>(a, chunks, per, acc, m, l, s);
    case 64: return kvflash_launch_tile<64>(a, chunks, per, acc, m, l, s);
    case 96: return kvflash_launch_tile<96>(a, chunks, per, acc, m, l, s);
    case 128: return kvflash_launch_tile<128>(a, chunks, per, acc, m, l, s);
    default: return -1;
    }
}

#endif  /* SPITE_QWEN3_KV_ATTN_FLASH_SM120_INL */

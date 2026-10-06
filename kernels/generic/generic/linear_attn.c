/*
 * kernels/generic/generic/linear_attn.c
 *
 * Reference Gated Delta Net layer, one decode token (ABI v7 `linear_attn`):
 *
 *     out += W_out . gated_norm(core(x))
 *
 * including the input projections.  Ground truth for every GPU implementation;
 * the maths is documented at SpiteGdnFn in core/abi.h and was checked against
 * llama.cpp's qwen35 graph (build_layer_attn_linear + delta-net autoregressive).
 *
 * Weights may be any SpiteType (streamed through ref_gemv, never expanded);
 * every reduction accumulates in double and is rounded to float only where the
 * value is stored (projections, conv output, q/k, core, state, out), which is
 * where any F32 implementation rounds too.
 *
 * Validation happens before the first write: a bad tensor / geometry returns -1,
 * a too-small ctx->scratchpad returns -2, and in neither case are `out`,
 * `conv_hist` or `state` touched.  Scratch layout is that of
 * spite_gdn_scratch_floats(): qkv | z | core | beta | alpha.
 *
 * conv_hist layout (private to the kernel by the ABI; this is the reference's):
 * [K-1][C] floats, row 0 the oldest input.
 */

#include <math.h>
#include <stdint.h>
#include "../../../core/abi.h"
#include "ref_common.h"

static double silu_d(double v) { return v / (1.0 + exp(-v)); }

int spite_generic_linear_attn(
    SpiteTensor          *out,
    const SpiteTensor    *x,
    const SpiteTensor    *w_qkv,
    const SpiteTensor    *w_gate,
    const SpiteTensor    *w_beta,
    const SpiteTensor    *w_alpha,
    const SpiteTensor    *w_out,
    const SpiteTensor    *conv_w,
    const SpiteTensor    *ssm_dt,
    const SpiteTensor    *ssm_a,
    const SpiteTensor    *ssm_norm,
    SpiteTensor          *conv_hist,
    SpiteTensor          *state,
    const SpiteGdnParams *p,
    const SpiteCtx       *ctx
) {
    if (!out || !x || !w_qkv || !w_gate || !w_beta || !w_alpha || !w_out || !conv_w || !ssm_dt ||
        !ssm_a || !ssm_norm || !conv_hist || !state || !p || !ctx)
        return -1;
    if (p->n_kh < 1 || p->n_vh < 1 || p->head_dim < 1 || p->d_conv < 1 ||
        p->n_kh > REF_MAX_DIM || p->n_vh > REF_MAX_DIM || p->head_dim > REF_MAX_DIM ||
        p->d_conv > REF_MAX_DIM || p->n_vh % p->n_kh != 0)
        return -1;

    const int64_t nkh = p->n_kh, nvh = p->n_vh, S = p->head_dim, K = p->d_conv;
    const int64_t kd = nkh * S, vd = nvh * S;
    const int64_t C = 2 * kd + vd;                    /* conv channels: q | k | v */
    const int64_t d_model = (int64_t)x->ne[0];
    const int64_t d_out = (int64_t)out->ne[0];

    if (d_model < 1 || d_out < 1) return -1;
    if (!ref_f32_vec(x, d_model) || !ref_f32_vec(out, d_out)) return -1;
    if (!ref_weight_ok(w_qkv, d_model, C) || !ref_weight_ok(w_gate, d_model, vd) ||
        !ref_weight_ok(w_beta, d_model, nvh) || !ref_weight_ok(w_alpha, d_model, nvh) ||
        !ref_weight_ok(w_out, vd, d_out))
        return -1;
    if (!ref_f32_vec(conv_w, K * C) || !ref_f32_vec(ssm_dt, nvh) || !ref_f32_vec(ssm_a, nvh) ||
        !ref_f32_vec(ssm_norm, S) || !ref_f32_vec(state, nvh * S * S))
        return -1;
    if (K > 1 && !ref_f32_vec(conv_hist, (K - 1) * C)) return -1;   /* K == 1: no history */

    int owned = 0, err = -1;
    float *buf = ref_work_buf(ctx, spite_gdn_scratch_floats(p), &owned, &err);
    if (!buf) return err;

    float *qkv   = buf;                  /* C: projected, then conv+SiLU in place (q | k | v) */
    float *z     = qkv + C;              /* V */
    float *core  = z + vd;               /* V: delta-rule output, then the gated-norm result */
    float *beta  = core + vd;            /* n_vh */
    float *alpha = beta + nvh;           /* n_vh */

    const float *xin = (const float *)x->data;
    const float *cw = (const float *)conv_w->data;
    const float *dt = (const float *)ssm_dt->data;
    const float *am = (const float *)ssm_a->data;
    const float *nw = (const float *)ssm_norm->data;
    float *hist = (float *)conv_hist->data;
    float *M = (float *)state->data;

    ref_gemv(w_qkv, xin, qkv, 0, 1);
    ref_gemv(w_gate, xin, z, 0, 1);
    ref_gemv(w_beta, xin, beta, 0, 1);
    ref_gemv(w_alpha, xin, alpha, 0, 1);

    /* Depthwise causal conv over [hist (K-1, oldest first) | qkv], then SiLU;
     * the history then drops its oldest row and appends this raw input. */
    for (int64_t c = 0; c < C; c++) {
        const float *wc = cw + c * K;                 /* tap k of channel c at conv_w[c*K + k] */
        double acc = (double)qkv[c] * (double)wc[K - 1];
        for (int64_t i = 0; i < K - 1; i++) acc += (double)hist[i * C + c] * (double)wc[i];
        for (int64_t i = 0; i + 1 < K - 1; i++) hist[i * C + c] = hist[(i + 1) * C + c];
        if (K > 1) hist[(K - 2) * C + c] = qkv[c];
        qkv[c] = (float)silu_d(acc);
    }

    /* l2norm(x) = x / sqrt(mean(x^2) + eps/S) / sqrt(S) on every q and k head. */
    float *q = qkv, *k = qkv + kd, *v = qkv + 2 * kd;
    const double eps = (double)p->norm_eps;
    for (int64_t i = 0; i < 2 * nkh; i++) {
        float *head = q + i * S;                      /* q heads then k heads are contiguous */
        double ss = 0.0;
        for (int64_t s = 0; s < S; s++) ss += (double)head[s] * (double)head[s];
        const double den = sqrt(ss / (double)S + eps / (double)S);
        for (int64_t s = 0; s < S; s++)
            head[s] = (float)((double)head[s] / den / sqrt((double)S));
    }

    /* Delta rule per value head (kh = vh % n_kh):
     *   M *= exp(g); d = (v - M^T k) * beta; M += k (x) d; o = M^T (q / sqrt(S))
     * Every output column s depends only on column s of M, so one pass per s. */
    const double qscale = 1.0 / sqrt((double)S);
    for (int64_t vh = 0; vh < nvh; vh++) {
        const double zz = (double)alpha[vh] + (double)dt[vh];
        const double g = (zz > 20.0 ? zz : log1p(exp(zz))) * (double)am[vh];  /* softplus * A */
        const double decay = exp(g);
        const double b = 1.0 / (1.0 + exp(-(double)beta[vh]));
        const float *qh = q + (vh % nkh) * S;
        const float *kh = k + (vh % nkh) * S;
        const float *vv = v + vh * S;
        float *Mh = M + vh * S * S;                   /* M[r][s] at Mh[r*S + s] */
        float *o = core + vh * S;

        for (int64_t s = 0; s < S; s++) {
            double sk = 0.0;
            for (int64_t r = 0; r < S; r++) {
                const float m = (float)((double)Mh[r * S + s] * decay);
                Mh[r * S + s] = m;
                sk += (double)m * (double)kh[r];
            }
            const double d = ((double)vv[s] - sk) * b;
            double os = 0.0;
            for (int64_t r = 0; r < S; r++) {
                const float m = (float)((double)Mh[r * S + s] + (double)kh[r] * d);
                Mh[r * S + s] = m;
                os += (double)m * ((double)qh[r] * qscale);
            }
            o[s] = (float)os;
        }
    }

    /* y[vh*S + s] = rms_norm(o[vh], ssm_norm, eps)[s] * silu(z[vh*S + s]) */
    for (int64_t vh = 0; vh < nvh; vh++) {
        float *o = core + vh * S;
        double ss = 0.0;
        for (int64_t s = 0; s < S; s++) ss += (double)o[s] * (double)o[s];
        const double inv = 1.0 / sqrt(ss / (double)S + eps);
        for (int64_t s = 0; s < S; s++)
            o[s] = (float)((double)o[s] * inv * (double)nw[s] * silu_d((double)z[vh * S + s]));
    }

    ref_gemv(w_out, core, (float *)out->data, 1, 1);

    if (owned) free(buf);
    return 0;
}

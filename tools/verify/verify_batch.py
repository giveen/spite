#!/usr/bin/env python3
"""
tools/verify/verify_batch.py — check a kernel's *batched* ops (prefill) against
the single-token path.

    python3 tools/verify/verify_batch.py <kernel.so>

A batched op takes the activation as `[cols, m]` (one column per token) instead
of `[cols]`. For the reference the batched result must equal m single-token
calls with the KV cache / recurrent state carried between them — bit for bit
where the kernel keeps the same arithmetic order (it does for the generic
reference).

Covers `matmul` and `ffn` against a float64 oracle, and `attention_ex` and
`linear_attn` against their own sequential m=1 runs. Ops the kernel leaves NULL
are SKIP. Exit code 0 = all checks passed.
"""

import ctypes
import math
import random
import sys

F32 = 0

class SpiteTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("ne", ctypes.c_uint32 * 4),
                ("nb", ctypes.c_uint64 * 4), ("kind", ctypes.c_uint32)]

class SpiteCtx(ctypes.Structure):
    _fields_ = [("n_ctx", ctypes.c_int), ("n_batch", ctypes.c_int),
                ("n_threads", ctypes.c_int), ("pos", ctypes.c_int),
                ("n_heads", ctypes.c_int), ("n_kv_heads", ctypes.c_int),
                ("gpu_stream", ctypes.c_void_p), ("scratchpad", ctypes.c_void_p),
                ("scratchpad_bytes", ctypes.c_size_t)]

class SpiteKvCache(ctypes.Structure):
    _fields_ = [("k", SpiteTensor), ("v", SpiteTensor), ("layer", ctypes.c_int)]

class GdnParams(ctypes.Structure):
    _fields_ = [("n_kh", ctypes.c_int32), ("n_vh", ctypes.c_int32),
                ("head_dim", ctypes.c_int32), ("d_conv", ctypes.c_int32),
                ("norm_eps", ctypes.c_float)]

class AttnParams(ctypes.Structure):
    _fields_ = [("head_dim", ctypes.c_int32), ("rope_dim", ctypes.c_int32),
                ("gated_q", ctypes.c_int32)]

def arr(vals):
    a = (ctypes.c_float * len(vals))()
    for i, v in enumerate(vals):
        a[i] = v
    return a

def tensor(buf, ne):
    ne = list(ne) + [1] * (4 - len(ne))
    nb = [4, 0, 0, 0]
    for i in range(1, 4):
        nb[i] = nb[i - 1] * max(ne[i - 1], 1)
    t = SpiteTensor()
    t.data = ctypes.cast(buf, ctypes.c_void_p)
    t.ne[:] = ne
    t.nb[:] = nb
    t.kind = F32
    return t

def rand(n, r):
    return [r.uniform(-1, 1) for _ in range(n)]

def zeros(n):
    return [0.0] * n

def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

def silu(x):
    return x / (1.0 + math.exp(-x))

def main(path):
    lib = ctypes.CDLL(path)
    rng = random.Random(11)
    ok = True

    # ── matmul: out[r,t] = sum_c w[c,r] * x[c,t] ──────────────────────────
    if not hasattr(lib, "spite_generic_matmul"):
        print("matmul: SKIP (not the generic kernel)")
    else:
        lib.spite_generic_matmul.restype = ctypes.c_int
        C, R, m = 5, 7, 3
        w, x = rand(C * R, rng), rand(C * m, rng)
        ob = arr(zeros(R * m))
        ctx = SpiteCtx(0, m, 1, 0, 0, 0, None, None, 0)
        ret = lib.spite_generic_matmul(ctypes.byref(tensor(ob, [R, m])),
                                       ctypes.byref(tensor(arr(x), [C, m])),
                                       ctypes.byref(tensor(arr(w), [C, R])),
                                       ctypes.byref(ctx))
        ref = [sum(w[c + r * C] * x[c + t * C] for c in range(C))
               for t in range(m) for r in range(R)]
        d = maxdiff(list(ob), ref) if ret == 0 else float("inf")
        print(f"matmul m={m}: ret={ret} max_abs_diff={d:.2e}")
        ok &= ret == 0 and d < 1e-5

    # ── ffn: gate/up [hidden,ffn], down [ffn,hidden], out += ──────────────
    if not hasattr(lib, "spite_generic_ffn"):
        print("ffn: SKIP")
    else:
        lib.spite_generic_ffn.restype = ctypes.c_int
        hidden, ffn, m = 6, 5, 3
        wg, wu = rand(hidden * ffn, rng), rand(hidden * ffn, rng)
        wd = rand(ffn * hidden, rng)
        x, out0 = rand(hidden * m, rng), rand(hidden * m, rng)
        bout = arr(out0)
        ctx = SpiteCtx(0, m, 1, 0, 0, 0, None, None, 0)
        ret = lib.spite_generic_ffn(ctypes.byref(tensor(bout, [hidden, m])),
                                    ctypes.byref(tensor(arr(x), [hidden, m])),
                                    ctypes.byref(tensor(arr(wg), [hidden, ffn])),
                                    ctypes.byref(tensor(arr(wu), [hidden, ffn])),
                                    ctypes.byref(tensor(arr(wd), [ffn, hidden])),
                                    0, ctypes.byref(ctx))
        ref = list(out0)
        for t in range(m):
            g = [sum(wg[c + i * hidden] * x[c + t * hidden] for c in range(hidden)) for i in range(ffn)]
            u = [sum(wu[c + i * hidden] * x[c + t * hidden] for c in range(hidden)) for i in range(ffn)]
            g = [silu(g[i]) * u[i] for i in range(ffn)]
            for j in range(hidden):
                ref[j + t * hidden] += sum(wd[i + j * ffn] * g[i] for i in range(ffn))
        d = maxdiff(list(bout), ref) if ret == 0 else float("inf")
        print(f"ffn m={m}: ret={ret} max_abs_diff={d:.2e}")
        ok &= ret == 0 and d < 1e-5

    # ── attention_ex: batched vs sequential ───────────────────────────────
    if not hasattr(lib, "spite_generic_attention_ex"):
        print("attention_ex: SKIP")
    else:
        lib.spite_generic_attention_ex.restype = ctypes.c_int
        nh, nkv, hd, rd, gated = 4, 2, 8, 8, 1
        qs = 2 * hd if gated else hd
        dm, nctx, m = 16, 16, 3
        wq, wk = rand(dm * nh * qs, rng), rand(dm * nkv * hd, rng)
        wv, wo = rand(dm * nkv * hd, rng), rand(nh * hd * dm, rng)
        x = rand(dm * m, rng)
        p = AttnParams(hd, rd, gated)
        Wq, Wk, Wv, Wo = arr(wq), arr(wk), arr(wv), arr(wo)

        def attn(pos, mm, kb, vb):
            kv = SpiteKvCache(tensor(kb, [nkv * hd, nctx]), tensor(vb, [nkv * hd, nctx]), 0)
            xb = arr(x[: dm * mm]) if mm == m else arr(x[dm * pos: dm * (pos + 1)])
            ob = arr(zeros(dm * mm))
            ctx = SpiteCtx(nctx, mm, 1, pos, nh, nkv, None, None, 0)
            r = lib.spite_generic_attention_ex(
                ctypes.byref(tensor(ob, [dm, mm])), ctypes.byref(tensor(xb, [dm, mm])),
                ctypes.byref(tensor(Wq, [dm, nh * qs])), ctypes.byref(tensor(Wk, [dm, nkv * hd])),
                ctypes.byref(tensor(Wv, [dm, nkv * hd])), ctypes.byref(tensor(Wo, [nh * hd, dm])),
                None, None, ctypes.c_float(1e-5), ctypes.byref(kv), ctypes.c_float(10000.0),
                ctypes.byref(p), ctypes.byref(ctx))
            assert r == 0, r
            return list(ob)

        b = attn(0, m, arr(zeros(nkv * hd * nctx)), arr(zeros(nkv * hd * nctx)))
        kb, vb = arr(zeros(nkv * hd * nctx)), arr(zeros(nkv * hd * nctx))
        seq = []
        for t in range(m):
            seq += attn(t, 1, kb, vb)
        d = maxdiff(b, seq)
        print(f"attention_ex m={m} vs sequential: max_abs_diff={d:.2e}")
        ok &= d < 1e-6

    # ── linear_attn: batched vs sequential ────────────────────────────────
    if not hasattr(lib, "spite_generic_linear_attn"):
        print("linear_attn: SKIP")
    else:
        lib.spite_generic_linear_attn.restype = ctypes.c_int
        nkh, nvh, S, K, dm, dout, m = 2, 4, 8, 3, 16, 16, 3
        C = 2 * nkh * S + nvh * S
        vd = nvh * S
        wqkv, wg, wb = rand(dm * C, rng), rand(dm * vd, rng), rand(dm * nvh, rng)
        wa, wout = rand(dm * nvh, rng), rand(vd * dout, rng)
        cw, dt, am, nw = rand(K * C, rng), rand(nvh, rng), rand(nvh, rng), rand(S, rng)
        xg = rand(dm * m, rng)
        gp = GdnParams(nkh, nvh, S, K, 1e-5)
        Wqkv, Wg, Wb, Wa, Wout = arr(wqkv), arr(wg), arr(wb), arr(wa), arr(wout)
        Cw, Dt, Am, Nw = arr(cw), arr(dt), arr(am), arr(nw)

        def gdn(pos, mm, hist, st):
            ob = arr(zeros(dout * mm))
            xb = arr(xg[dm * pos: dm * (pos + mm)])
            ctx = SpiteCtx(64, mm, 1, pos, 0, 0, None, None, 0)
            r = lib.spite_generic_linear_attn(
                ctypes.byref(tensor(ob, [dout, mm])), ctypes.byref(tensor(xb, [dm, mm])),
                ctypes.byref(tensor(Wqkv, [dm, C])), ctypes.byref(tensor(Wg, [dm, vd])),
                ctypes.byref(tensor(Wb, [dm, nvh])), ctypes.byref(tensor(Wa, [dm, nvh])),
                ctypes.byref(tensor(Wout, [vd, dout])), ctypes.byref(tensor(Cw, [K * C, 1])),
                ctypes.byref(tensor(Dt, [nvh, 1])), ctypes.byref(tensor(Am, [nvh, 1])),
                ctypes.byref(tensor(Nw, [S, 1])), ctypes.byref(tensor(hist, [(K - 1) * C, 1])),
                ctypes.byref(tensor(st, [nvh * S * S, 1])), ctypes.byref(gp), ctypes.byref(ctx))
            assert r == 0, r
            return list(ob)

        b = gdn(0, m, arr(zeros((K - 1) * C)), arr(zeros(nvh * S * S)))
        h2, s2 = arr(zeros((K - 1) * C)), arr(zeros(nvh * S * S))
        seq = []
        for t in range(m):
            seq += gdn(t, 1, h2, s2)
        d = maxdiff(b, seq)
        print(f"linear_attn m={m} vs sequential: max_abs_diff={d:.2e}")
        ok &= d < 1e-6

    print("BATCH PASSED" if ok else "BATCH FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

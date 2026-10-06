#!/usr/bin/env python3
"""
tools/verify/verify_batch_cuda.py — check a CUDA kernel's *batched* ffn.

    LD_LIBRARY_PATH=<dir with libcudart.so.12> \
    python3 tools/verify/verify_batch_cuda.py <kernel.so> <libcudart.so.12>

Runs the kernel's own `.ffn` pointer with a device activation `[hidden, m]`
(m > 1) and a device scratchpad, and compares to a float64 reference. This
exercises the batched (multi-column) GEMV path. A kernel without `ffn` is SKIP.
Exit code 0 = passed.
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

class SpiteKernelInfo(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("model_arch", ctypes.c_char_p),
                ("gpu_arch", ctypes.c_char_p), ("author", ctypes.c_char_p),
                ("supported_quants", ctypes.c_uint32 * 8),
                ("rms_norm", ctypes.c_void_p), ("attention", ctypes.c_void_p),
                ("mla", ctypes.c_void_p), ("ffn", ctypes.c_void_p),
                ("layer", ctypes.c_void_p), ("speculative_verify", ctypes.c_void_p),
                ("prefill", ctypes.c_void_p), ("matmul", ctypes.c_void_p),
                ("kv_cache_kinds", ctypes.c_void_p), ("linear_attn", ctypes.c_void_p),
                ("attention_ex", ctypes.c_void_p), ("mtp_stem", ctypes.c_void_p),
                ("moe_ffn", ctypes.c_void_p)]

class SpiteKvCache(ctypes.Structure):
    _fields_ = [("k", SpiteTensor), ("v", SpiteTensor), ("layer", ctypes.c_int)]

class AttnParams(ctypes.Structure):
    _fields_ = [("head_dim", ctypes.c_int32), ("rope_dim", ctypes.c_int32),
                ("gated_q", ctypes.c_int32)]

def tensor(p, ne):
    ne = list(ne) + [1] * (4 - len(ne))
    nb = [4, 0, 0, 0]
    for i in range(1, 4):
        nb[i] = nb[i - 1] * max(ne[i - 1], 1)
    t = SpiteTensor()
    t.data = p
    t.ne[:] = ne
    t.nb[:] = nb
    t.kind = F32
    return t

def main(kernel_path, cudart_path):
    cudart = ctypes.CDLL(cudart_path)
    for fn in ("cudaMalloc", "cudaMemcpy", "cudaFree", "cudaDeviceSynchronize"):
        getattr(cudart, fn).restype = ctypes.c_int
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    lib = ctypes.CDLL(kernel_path)
    lib.spite_kernel_info.restype = ctypes.POINTER(SpiteKernelInfo)
    info = lib.spite_kernel_info().contents
    if not info.ffn:
        print("ffn: SKIP (kernel has no ffn)")
        return 0

    def alloc(nbytes):
        p = ctypes.c_void_p()
        assert cudart.cudaMalloc(ctypes.byref(p), nbytes) == 0
        return p

    def h2d(p, vals):
        a = (ctypes.c_float * len(vals))(*vals)
        assert cudart.cudaMemcpy(p, ctypes.cast(a, ctypes.c_void_p), ctypes.sizeof(a), 1) == 0

    def d2h(p, n):
        a = (ctypes.c_float * n)()
        assert cudart.cudaMemcpy(ctypes.cast(a, ctypes.c_void_p), p, n * 4, 2) == 0
        return list(a)

    FfnFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(SpiteTensor),
                             ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
                             ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
                             ctypes.c_int, ctypes.POINTER(SpiteCtx))
    ffn = FfnFn(info.ffn)

    rng = random.Random(3)
    hidden, fd, m = 6, 5, 3
    wg = [rng.uniform(-1, 1) for _ in range(hidden * fd)]
    wu = [rng.uniform(-1, 1) for _ in range(hidden * fd)]
    wd = [rng.uniform(-1, 1) for _ in range(fd * hidden)]
    x = [rng.uniform(-1, 1) for _ in range(hidden * m)]
    out0 = [rng.uniform(-1, 1) for _ in range(hidden * m)]

    pwg, pwu, pwd = alloc(hidden * fd * 4), alloc(hidden * fd * 4), alloc(fd * hidden * 4)
    px, pout = alloc(hidden * m * 4), alloc(hidden * m * 4)
    pscratch = alloc(2 * fd * m * 4)
    h2d(pwg, wg)
    h2d(pwu, wu)
    h2d(pwd, wd)
    h2d(px, x)
    h2d(pout, out0)

    ctx = SpiteCtx(0, m, 1, 0, 0, 0, None, ctypes.cast(pscratch, ctypes.c_void_p), 2 * fd * m * 4)
    ret = ffn(ctypes.byref(tensor(pout, [hidden, m])), ctypes.byref(tensor(px, [hidden, m])),
              ctypes.byref(tensor(pwg, [hidden, fd])), ctypes.byref(tensor(pwu, [hidden, fd])),
              ctypes.byref(tensor(pwd, [fd, hidden])), 0, ctypes.byref(ctx))
    assert cudart.cudaDeviceSynchronize() == 0
    got = d2h(pout, hidden * m)

    silu = lambda v: v / (1.0 + math.exp(-v))
    ref = list(out0)
    for t in range(m):
        g = [sum(wg[c + i * hidden] * x[c + t * hidden] for c in range(hidden)) for i in range(fd)]
        u = [sum(wu[c + i * hidden] * x[c + t * hidden] for c in range(hidden)) for i in range(fd)]
        g = [silu(g[i]) * u[i] for i in range(fd)]
        for j in range(hidden):
            ref[j + t * hidden] += sum(wd[i + j * fd] * g[i] for i in range(fd))
    d = max(abs(a - b) for a, b in zip(got, ref))
    print(f"ffn batch m={m}: ret={ret} max_abs_diff={d:.2e}")
    ok = ret == 0 and d < 1e-5

    if info.matmul:
        MatFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(SpiteTensor),
                                 ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
                                 ctypes.POINTER(SpiteCtx))
        mm = MatFn(info.matmul)
        C, R, k = 5, 7, 3
        w = [rng.uniform(-1, 1) for _ in range(C * R)]
        xx = [rng.uniform(-1, 1) for _ in range(C * k)]
        pW, pX, pO = alloc(C * R * 4), alloc(C * k * 4), alloc(R * k * 4)
        h2d(pW, w)
        h2d(pX, xx)
        ctx2 = SpiteCtx(0, k, 1, 0, 0, 0, None, None, 0)
        ret2 = mm(ctypes.byref(tensor(pO, [R, k])), ctypes.byref(tensor(pX, [C, k])),
                  ctypes.byref(tensor(pW, [C, R])), ctypes.byref(ctx2))
        assert cudart.cudaDeviceSynchronize() == 0
        got2 = d2h(pO, R * k)
        ref2 = [sum(w[c + r * C] * xx[c + t * C] for c in range(C))
                for t in range(k) for r in range(R)]
        d2 = max(abs(a - b) for a, b in zip(got2, ref2))
        print(f"matmul batch m={k}: ret={ret2} max_abs_diff={d2:.2e}")
        ok = ok and ret2 == 0 and d2 < 1e-5
    else:
        print("matmul: SKIP")

    if info.attention_ex:
        AttnExFn = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
            ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
            ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
            ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
            ctypes.c_float, ctypes.POINTER(SpiteKvCache), ctypes.c_float,
            ctypes.POINTER(AttnParams), ctypes.POINTER(SpiteCtx))
        attn = AttnExFn(info.attention_ex)
        nh, nkv, hd, rd, gated = 4, 2, 64, 64, 1
        dm, nctx, m = 64, 16, 3
        q_rows = nh * hd * (2 if gated else 1)
        kv_stride = nkv * hd
        wq = [rng.uniform(-1, 1) for _ in range(dm * q_rows)]
        wk = [rng.uniform(-1, 1) for _ in range(dm * kv_stride)]
        wv = [rng.uniform(-1, 1) for _ in range(dm * kv_stride)]
        wo = [rng.uniform(-1, 1) for _ in range(nh * hd * dm)]
        x = [rng.uniform(-1, 1) for _ in range(dm * m)]
        p = AttnParams(hd, rd, gated)
        pWq, pWk, pWv, pWo = alloc(dm * q_rows * 4), alloc(dm * kv_stride * 4), alloc(dm * kv_stride * 4), alloc(nh * hd * dm * 4)
        h2d(pWq, wq); h2d(pWk, wk); h2d(pWv, wv); h2d(pWo, wo)
        # KV rows must be 16-byte aligned with nb[1] = kv_stride*4.
        pK, pV = alloc(kv_stride * nctx * 4), alloc(kv_stride * nctx * 4)
        scratch_bytes = 1 << 21
        pScr = alloc(scratch_bytes)

        def run_attn(pos, mm, kbuf, vbuf, xslice):
            kv = SpiteKvCache(tensor(kbuf, [kv_stride, nctx]), tensor(vbuf, [kv_stride, nctx]), 0)
            pX, pO = alloc(dm * mm * 4), alloc(dm * mm * 4)
            h2d(pX, xslice)
            ctx = SpiteCtx(nctx, mm, 1, pos, nh, nkv, None, ctypes.cast(pScr, ctypes.c_void_p), scratch_bytes)
            r = attn(ctypes.byref(tensor(pO, [dm, mm])), ctypes.byref(tensor(pX, [dm, mm])),
                     ctypes.byref(tensor(pWq, [dm, q_rows])), ctypes.byref(tensor(pWk, [dm, kv_stride])),
                     ctypes.byref(tensor(pWv, [dm, kv_stride])), ctypes.byref(tensor(pWo, [nh * hd, dm])),
                     None, None, ctypes.c_float(1e-5), ctypes.byref(kv), ctypes.c_float(10000.0),
                     ctypes.byref(p), ctypes.byref(ctx))
            assert cudart.cudaDeviceSynchronize() == 0
            assert r == 0, r
            return d2h(pO, dm * mm)

        batched = run_attn(0, m, pK, pV, x)
        seq = []
        for t in range(m):
            seq += run_attn(t, 1, pK, pV, x[dm * t:dm * (t + 1)])
        d3 = max(abs(a - b) for a, b in zip(batched, seq))
        print(f"attention_ex batch m={m} vs sequential: max_abs_diff={d3:.2e}")
        ok = ok and d3 < 1e-5
    else:
        print("attention_ex: SKIP")

    print("BATCH CUDA PASSED" if ok else "BATCH CUDA FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

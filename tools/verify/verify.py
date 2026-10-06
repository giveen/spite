#!/usr/bin/env python3
"""
tools/verify/verify.py — verify a kernel .so against the generic reference

Usage:
    python3 tools/verify/verify.py <kernel.so> [--model <model.gguf>] [--ref <ref_kernel.so>]

Loads the kernel with ctypes, loads the generic reference kernel, runs each
op with randomized inputs, and compares outputs. Any element-wise
max error > 1e-4 is a failure.

Covers rms_norm, matmul (every SpiteType), ffn, attention, and the ABI v7 layer ops
`linear_attn` (Gated Delta Net layer) and `attention_ex` (partial RoPE + gated Q); the
generic reference is itself checked against pure-python float64 models of both layer ops.
An op a kernel leaves NULL is reported as SKIP.

Also performs an ABI version check: refuses kernels reporting a different
SPITE_ABI_VERSION from what core/abi.h reports.

Exit code 0 = all checks passed.
"""

import argparse
import ctypes
import os
import struct
import sys
import random
import math
import operator
from array import array

# ── ABI constants (must match core/abi.h; type ids are the GGUF/ggml ids) ────────────────────────────────

SPITE_ABI_VERSION = 7

SPITE_TYPE_F32  = 0
SPITE_TYPE_F16  = 1
SPITE_TYPE_BF16 = 30
SPITE_TYPE_Q8_0 = 8
SPITE_TYPE_Q5_1 = 7
SPITE_TYPE_Q4_0 = 2
SPITE_TYPE_Q4_K = 12
SPITE_TYPE_Q5_K = 13
SPITE_TYPE_Q6_K = 14

SPITE_FFN_SILU_GATE = 0
SPITE_FFN_GELU_GATE = 1
SPITE_FFN_GELU      = 2
SPITE_FFN_RELU      = 3

# ── ctypes struct definitions ─────────────────────────────────────────────

class SpiteTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ne",   ctypes.c_uint32 * 4),
        ("nb",   ctypes.c_uint64 * 4),
        ("kind", ctypes.c_uint32),
    ]

class SpiteCtx(ctypes.Structure):
    _fields_ = [
        ("n_ctx",            ctypes.c_int),
        ("n_batch",          ctypes.c_int),
        ("n_threads",        ctypes.c_int),
        ("pos",              ctypes.c_int),
        ("n_heads",          ctypes.c_int),
        ("n_kv_heads",       ctypes.c_int),
        ("gpu_stream",       ctypes.c_void_p),
        ("scratchpad",       ctypes.c_void_p),
        ("scratchpad_bytes", ctypes.c_size_t),
    ]

class SpiteKvCache(ctypes.Structure):
    _fields_ = [
        ("k",     SpiteTensor),
        ("v",     SpiteTensor),
        ("layer", ctypes.c_int),
    ]

class SpiteKernelInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version",        ctypes.c_uint32),
        ("model_arch",         ctypes.c_char_p),
        ("gpu_arch",           ctypes.c_char_p),
        ("author",             ctypes.c_char_p),
        ("supported_quants",   ctypes.c_uint32 * 8),
        ("rms_norm",           ctypes.c_void_p),
        ("attention",          ctypes.c_void_p),
        ("mla",                ctypes.c_void_p),
        ("ffn",                ctypes.c_void_p),
        ("layer",              ctypes.c_void_p),
        ("speculative_verify", ctypes.c_void_p),
        ("prefill",            ctypes.c_void_p),
        ("matmul",             ctypes.c_void_p),
        ("kv_cache_kinds",     ctypes.c_void_p),  # trailing optional ABI slot (v4)
        ("linear_attn",        ctypes.c_void_p),  # GDN layer (v7)
        ("attention_ex",       ctypes.c_void_p),  # partial RoPE + gated Q (v7)
    ]

class SpiteGdnParams(ctypes.Structure):
    _fields_ = [
        ("n_kh",     ctypes.c_int32),
        ("n_vh",     ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("d_conv",   ctypes.c_int32),
        ("norm_eps", ctypes.c_float),
    ]

class SpiteAttnParams(ctypes.Structure):
    _fields_ = [
        ("head_dim", ctypes.c_int32),
        ("rope_dim", ctypes.c_int32),
        ("gated_q",  ctypes.c_int32),
    ]

# ── Function Signatures ───────────────────────────────────────────────────

RmsNormFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.c_float,
    ctypes.POINTER(SpiteCtx),
)

MatmulFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteCtx),
)

FfnFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.c_uint32,
    ctypes.POINTER(SpiteCtx),
)

_TP = ctypes.POINTER(SpiteTensor)

# out, x, w_qkv, w_gate, w_beta, w_alpha, w_out, conv_w, ssm_dt, ssm_a, ssm_norm, conv_hist, state
GdnFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    *([_TP] * 13), ctypes.POINTER(SpiteGdnParams), ctypes.POINTER(SpiteCtx),
)

AttentionFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.c_float, ctypes.POINTER(SpiteKvCache),
    ctypes.c_float, ctypes.POINTER(SpiteCtx),
)

# out, x, wq, wk, wv, wo, q_norm, k_norm, norm_eps, kvcache, rope_freq_base, params, ctx
AttentionExFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    *([_TP] * 8), ctypes.c_float, ctypes.POINTER(SpiteKvCache),
    ctypes.c_float, ctypes.POINTER(SpiteAttnParams), ctypes.POINTER(SpiteCtx),
)

# ── Strides and Helpers ───────────────────────────────────────────────────

# (SpiteType id, name, bytes per block, elements per block) for every SpiteType in
# core/abi.h — ids are the GGUF/ggml tensor type ids.  Full-precision types have
# block size 1.  Must match spite_type_block_bytes/elements().
SPITE_TYPES = [
    (0, "F32", 4, 1), (1, "F16", 2, 1), (30, "BF16", 2, 1),
    (2, "Q4_0", 18, 32), (3, "Q4_1", 20, 32), (6, "Q5_0", 22, 32), (7, "Q5_1", 24, 32),
    (8, "Q8_0", 34, 32), (41, "Q1_0", 18, 128), (42, "Q2_0", 18, 64),
    (10, "Q2_K", 84, 256), (11, "Q3_K", 110, 256), (12, "Q4_K", 144, 256),
    (13, "Q5_K", 176, 256), (14, "Q6_K", 210, 256),
    (16, "IQ2_XXS", 66, 256), (17, "IQ2_XS", 74, 256), (22, "IQ2_S", 82, 256),
    (18, "IQ3_XXS", 98, 256), (21, "IQ3_S", 110, 256), (19, "IQ1_S", 50, 256),
    (29, "IQ1_M", 56, 256), (20, "IQ4_NL", 18, 32), (23, "IQ4_XS", 136, 256),
    (34, "TQ1_0", 54, 256), (35, "TQ2_0", 66, 256), (39, "MXFP4", 17, 32),
    (40, "NVFP4", 36, 64),
]
TYPE_NAME        = {t: n for t, n, _, _ in SPITE_TYPES}
TYPE_BLOCK_BYTES = {t: b for t, _, b, _ in SPITE_TYPES}
TYPE_BLOCK_ELEMS = {t: e for t, _, _, e in SPITE_TYPES}


def contiguous_strides(kind: int, ne: list[int] | tuple[int, ...]) -> list[int]:
    block_bytes = TYPE_BLOCK_BYTES.get(kind, 4)
    block_elements = TYPE_BLOCK_ELEMS.get(kind, 1)

    nb0 = block_bytes
    cols_blk = max(1, ne[0] // block_elements)
    nb1 = nb0 * cols_blk
    nb2 = nb1 * max(1, ne[1])
    nb3 = nb2 * max(1, ne[2])
    return [nb0, nb1, nb2, nb3]


def make_tensor(data_f32: list[float], ne: list[int] | tuple[int, ...] | None = None,
                kind: int = SPITE_TYPE_F32) -> tuple[SpiteTensor, ctypes.Array]:
    """Create a host SpiteTensor backed by a ctypes float array."""
    if ne is None:
        ne = [len(data_f32), 1, 1, 1]
    arr = (ctypes.c_float * len(data_f32))(*data_f32)
    t = SpiteTensor()
    t.data = ctypes.cast(arr, ctypes.c_void_p)
    for i in range(4):
        t.ne[i] = ne[i]
    strides = contiguous_strides(kind, ne)
    for i in range(4):
        t.nb[i] = strides[i]
    t.kind = kind
    return t, arr


def make_ctx() -> SpiteCtx:
    ctx = SpiteCtx()
    ctx.n_ctx     = 2048
    ctx.n_batch   = 1
    ctx.n_threads = 1
    return ctx


KV_BLOCK_BYTES = {
    SPITE_TYPE_F32: 4, SPITE_TYPE_F16: 2, SPITE_TYPE_BF16: 2,
    SPITE_TYPE_Q8_0: 34, SPITE_TYPE_Q5_1: 24, SPITE_TYPE_Q4_0: 18,
}
KV_BLOCK_ELEMS = {SPITE_TYPE_Q8_0: 32, SPITE_TYPE_Q5_1: 32, SPITE_TYPE_Q4_0: 32}
KV_TIER_NAMES = {SPITE_TYPE_F32: "F32", SPITE_TYPE_F16: "F16", SPITE_TYPE_Q8_0: "Q8_0",
                 SPITE_TYPE_Q5_1: "Q5_1", SPITE_TYPE_Q4_0: "Q4_0"}


def kv_row_bytes(kind: int, n_elem: int) -> int:
    """Bytes per KV row; mirrors kvq_row_bytes() in kv_attn.inl."""
    if kind in (SPITE_TYPE_F32, SPITE_TYPE_BF16):
        return n_elem * KV_BLOCK_BYTES[kind]
    if kind == SPITE_TYPE_F16:
        return n_elem * 2
    return ((n_elem + 31) // 32) * KV_BLOCK_BYTES[kind]


def kv_tier_blob(kind: int, rows: int, n_elem: int, rng: random.Random) -> bytes:
    """A KV cache image of `rows` rows at `kind`, in the project's block layout.

    The block fields are filled with ordinary magnitudes so both sides compute
    on finite inputs; any byte pattern would do for a differential comparison,
    but a NaN scale would fail the tolerance for a reason that is not a bug.
    """
    if kind == SPITE_TYPE_F32:
        return struct.pack(f"<{rows * n_elem}f", *[rng.gauss(0, 1.0) for _ in range(rows * n_elem)])
    if kind == SPITE_TYPE_F16:
        return struct.pack(f"<{rows * n_elem}e", *[rng.gauss(0, 1.0) for _ in range(rows * n_elem)])
    blocks = (n_elem + 31) // 32
    out = bytearray()
    for _ in range(rows * blocks):
        if kind == SPITE_TYPE_Q8_0:
            out += struct.pack("<e", rng.uniform(0.005, 0.05))
            out += bytes(rng.randrange(256) for _ in range(32))
        elif kind == SPITE_TYPE_Q5_1:
            out += struct.pack("<e", rng.uniform(0.001, 0.02))
            out += struct.pack("<e", rng.uniform(-0.3, 0.3))
            out += struct.pack("<I", rng.getrandbits(32))
            out += bytes(rng.randrange(256) for _ in range(16))
        else:
            out += struct.pack("<e", rng.uniform(0.005, 0.05))
            out += bytes(rng.randrange(256) for _ in range(16))
    assert len(out) == rows * kv_row_bytes(kind, n_elem)
    return bytes(out)


def looks_cuda(info: SpiteKernelInfo) -> bool:
    """Same heuristic main() uses to pick the device path for the test kernel."""
    arch = (info.gpu_arch or b"").lower()
    return b"sm_" in arch or b"cuda" in arch or b"nvidia" in arch


def declared_kv_tiers(info: SpiteKernelInfo) -> int:
    """KV tiers the kernel's attention op accepts (NULL slot = F32 only)."""
    if not info.kv_cache_kinds:
        return 1 << SPITE_TYPE_F32
    fn = ctypes.CFUNCTYPE(ctypes.c_uint64)(info.kv_cache_kinds)
    return fn()


def max_abs_diff(a: list[float], b: list[float]) -> float:
    """Max element-wise |a - b|, and inf if either side produced a NaN.

    A NaN must fail: `nan > 1e-4` is False, so returning the NaN straight from
    `max()` would let a kernel with an uninitialized accumulator pass.
    """
    err = 0.0
    for x, y in zip(a, b):
        if math.isnan(x) or math.isnan(y):
            return math.inf
        err = max(err, abs(x - y))
    return err


def load_kernel(path: str):
    lib = ctypes.CDLL(path)
    lib.spite_kernel_info.restype  = ctypes.POINTER(SpiteKernelInfo)
    lib.spite_kernel_info.argtypes = []
    return lib


class CudaHelper:
    def __init__(self):
        self.cudart = None
        try:
            self.cudart = ctypes.CDLL("libcudart.so")
            self.cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            self.cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            self.cudart.cudaFree.argtypes = [ctypes.c_void_p]
            self.cudart.cudaDeviceSynchronize.argtypes = []
        except Exception:
            self.cudart = None

    @property
    def available(self) -> bool:
        return self.cudart is not None

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        ret = self.cudart.cudaMalloc(ctypes.byref(ptr), nbytes)
        if ret != 0:
            raise RuntimeError(f"cudaMalloc failed with code {ret}")
        return ptr.value

    def free(self, ptr_val: int):
        if ptr_val:
            self.cudart.cudaFree(ctypes.c_void_p(ptr_val))

    def h2d(self, dst_val: int, src_arr, nbytes: int):
        ret = self.cudart.cudaMemcpy(ctypes.c_void_p(dst_val), ctypes.cast(src_arr, ctypes.c_void_p), nbytes, 1)
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed with code {ret}")

    def d2h(self, dst_arr, src_val: int, nbytes: int):
        ret = self.cudart.cudaMemcpy(ctypes.cast(dst_arr, ctypes.c_void_p), ctypes.c_void_p(src_val), nbytes, 2)
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy D2H failed with code {ret}")

    def sync(self):
        self.cudart.cudaDeviceSynchronize()


# ── ABI check ─────────────────────────────────────────────────────────────

def check_abi(lib, label: str) -> SpiteKernelInfo:
    info_ptr = lib.spite_kernel_info()
    if not info_ptr:
        print(f"  ERROR: {label} spite_kernel_info() returned NULL")
        sys.exit(1)
    info = info_ptr.contents
    if info.abi_version != SPITE_ABI_VERSION:
        print(f"  ERROR: {label} ABI version {info.abi_version} != expected {SPITE_ABI_VERSION}")
        sys.exit(1)
    print(f"  {label}: arch={info.model_arch.decode()} gpu={info.gpu_arch.decode()} "
          f"abi={info.abi_version} author={info.author.decode()}")
    return info


# ── Verification Functions ────────────────────────────────────────────────

def verify_rms_norm(ref_info: SpiteKernelInfo, test_info: SpiteKernelInfo,
                    cuda: CudaHelper, is_cuda: bool, dim: int = 64) -> bool:
    print(f"\n  [rms_norm] dim={dim}")
    if not ref_info.rms_norm:
        print("    SKIP: reference kernel has no rms_norm")
        return True
    if not test_info.rms_norm:
        print("    SKIP: test kernel has no rms_norm")
        return True

    rng = random.Random(42)
    x_data = [rng.gauss(0, 1) for _ in range(dim)]
    w_data = [rng.uniform(0.5, 1.5) for _ in range(dim)]
    out_ref_arr = (ctypes.c_float * dim)()

    tx, _ = make_tensor(x_data, [dim, 1, 1, 1])
    tw, _ = make_tensor(w_data, [dim, 1, 1, 1])
    to, _ = make_tensor([0.0] * dim, [dim, 1, 1, 1])
    to.data = ctypes.cast(out_ref_arr, ctypes.c_void_p)
    ctx_ref = make_ctx()

    ref_fn = RmsNormFn(ref_info.rms_norm)
    ret_ref = ref_fn(ctypes.byref(to), ctypes.byref(tx), ctypes.byref(tw), 1e-5, ctypes.byref(ctx_ref))
    if ret_ref != 0:
        print(f"    SKIP: reference rms_norm returned {ret_ref}")
        return True

    test_fn = RmsNormFn(test_info.rms_norm)
    if not is_cuda:
        out_test_arr = (ctypes.c_float * dim)()
        to_test, _ = make_tensor([0.0] * dim, [dim, 1, 1, 1])
        to_test.data = ctypes.cast(out_test_arr, ctypes.c_void_p)
        ctx_test = make_ctx()
        ret_test = test_fn(ctypes.byref(to_test), ctypes.byref(tx), ctypes.byref(tw), 1e-5, ctypes.byref(ctx_test))
        if ret_test != 0:
            print(f"    FAIL: test rms_norm returned {ret_test}")
            return False
        test_out = list(out_test_arr)
    else:
        if not cuda.available:
            print("    SKIP: CUDA runtime not available")
            return True
        nbytes = dim * 4
        d_x = cuda.malloc(nbytes)
        d_w = cuda.malloc(nbytes)
        d_o = cuda.malloc(nbytes)
        try:
            cuda.h2d(d_x, (ctypes.c_float * dim)(*x_data), nbytes)
            cuda.h2d(d_w, (ctypes.c_float * dim)(*w_data), nbytes)
            tx_gpu, _ = make_tensor([0.0] * dim, [dim, 1, 1, 1])
            tw_gpu, _ = make_tensor([0.0] * dim, [dim, 1, 1, 1])
            to_gpu, _ = make_tensor([0.0] * dim, [dim, 1, 1, 1])
            tx_gpu.data = d_x
            tw_gpu.data = d_w
            to_gpu.data = d_o
            ctx_gpu = make_ctx()
            ret_test = test_fn(ctypes.byref(to_gpu), ctypes.byref(tx_gpu), ctypes.byref(tw_gpu), 1e-5, ctypes.byref(ctx_gpu))
            cuda.sync()
            if ret_test != 0:
                print(f"    FAIL: test rms_norm returned {ret_test}")
                return False
            out_test_arr = (ctypes.c_float * dim)()
            cuda.d2h(out_test_arr, d_o, nbytes)
            test_out = list(out_test_arr)
        finally:
            cuda.free(d_x)
            cuda.free(d_w)
            cuda.free(d_o)

    err = max_abs_diff(list(out_ref_arr), test_out)
    if err > 1e-4:
        print(f"    FAIL: max_abs_diff={err:.2e} (threshold 1e-4)")
        return False
    print(f"    OK: max_abs_diff={err:.2e}")
    return True


def pack_q8_0(values: list[float]) -> bytes:
    """Pack a flat Q8_0 row into GGUF blocks: fp16 scale then 32 signed bytes."""
    assert len(values) % 32 == 0
    out = bytearray()
    for i in range(0, len(values), 32):
        group = values[i:i + 32]
        amax = max(abs(v) for v in group) or 1.0
        scale = amax / 127.0
        out += struct.pack("<e", scale)
        for v in group:
            out += bytes([max(-128, min(127, round(v / scale))) & 0xFF])
    return bytes(out)


def verify_matmul(ref_info: SpiteKernelInfo, test_info: SpiteKernelInfo,
                  cuda: CudaHelper, is_cuda: bool, cols: int = 64, rows: int = 128,
                  q8: bool = False) -> bool:
    tag = " q8_0" if q8 else ""
    print(f"\n  [matmul] shape=[{rows}, {cols}]{tag}")
    if not ref_info.matmul:
        print("    SKIP: reference kernel has no matmul")
        return True
    if not test_info.matmul:
        print("    SKIP: test kernel has no matmul")
        return True

    rng = random.Random(42)
    x_data = [rng.gauss(0, 1) for _ in range(cols)]
    w_scale = 1.0 / math.sqrt(cols)
    w_data = [rng.gauss(0, w_scale) for _ in range(rows * cols)]
    out_ref_arr = (ctypes.c_float * rows)()

    tx, _ = make_tensor(x_data, [cols, 1, 1, 1])
    to, _ = make_tensor([0.0] * rows, [rows, 1, 1, 1])
    to.data = ctypes.cast(out_ref_arr, ctypes.c_void_p)
    ctx_ref = make_ctx()
    if q8:
        # Both sides read the packed bytes, so this is a differential check of
        # the kernel's Q8_0 decode path against the reference's own dequant.
        assert cols % 32 == 0
        w_bytes = pack_q8_0(w_data)
        w_buf = ctypes.create_string_buffer(w_bytes)
        tw, _ = make_tensor([0.0] * (rows * cols), [cols, rows, 1, 1], kind=SPITE_TYPE_Q8_0)
        tw.data = ctypes.cast(w_buf, ctypes.c_void_p)
    else:
        w_bytes = None
        w_buf = None
        tw, _ = make_tensor(w_data, [cols, rows, 1, 1])

    ref_fn = MatmulFn(ref_info.matmul)
    ret_ref = ref_fn(ctypes.byref(to), ctypes.byref(tx), ctypes.byref(tw), ctypes.byref(ctx_ref))
    if ret_ref != 0:
        print(f"    SKIP: reference matmul returned {ret_ref}")
        return True

    test_fn = MatmulFn(test_info.matmul)
    if not is_cuda:
        out_test_arr = (ctypes.c_float * rows)()
        to_test, _ = make_tensor([0.0] * rows, [rows, 1, 1, 1])
        to_test.data = ctypes.cast(out_test_arr, ctypes.c_void_p)
        ctx_test = make_ctx()
        ret_test = test_fn(ctypes.byref(to_test), ctypes.byref(tx), ctypes.byref(tw), ctypes.byref(ctx_test))
        if ret_test != 0:
            print(f"    FAIL: test matmul returned {ret_test}")
            return False
        test_out = list(out_test_arr)
    else:
        if not cuda.available:
            print("    SKIP: CUDA runtime not available")
            return True
        d_x = cuda.malloc(cols * 4)
        d_w = cuda.malloc(len(w_bytes) if q8 else rows * cols * 4)
        d_o = cuda.malloc(rows * 4)
        try:
            cuda.h2d(d_x, (ctypes.c_float * cols)(*x_data), cols * 4)
            if q8:
                cuda.h2d(d_w, (ctypes.c_ubyte * len(w_bytes))(*w_bytes), len(w_bytes))
            else:
                cuda.h2d(d_w, (ctypes.c_float * (rows * cols))(*w_data), rows * cols * 4)
            tx_gpu, _ = make_tensor([0.0] * cols, [cols, 1, 1, 1])
            tw_gpu, _ = make_tensor([0.0] * (rows * cols), [cols, rows, 1, 1],
                                    kind=SPITE_TYPE_Q8_0 if q8 else SPITE_TYPE_F32)
            to_gpu, _ = make_tensor([0.0] * rows, [rows, 1, 1, 1])
            tx_gpu.data = d_x
            tw_gpu.data = d_w
            to_gpu.data = d_o
            ctx_gpu = make_ctx()
            ret_test = test_fn(ctypes.byref(to_gpu), ctypes.byref(tx_gpu), ctypes.byref(tw_gpu), ctypes.byref(ctx_gpu))
            cuda.sync()
            if ret_test != 0:
                print(f"    FAIL: test matmul returned {ret_test}")
                return False
            out_test_arr = (ctypes.c_float * rows)()
            cuda.d2h(out_test_arr, d_o, rows * 4)
            test_out = list(out_test_arr)
        finally:
            cuda.free(d_x)
            cuda.free(d_w)
            cuda.free(d_o)

    err = max_abs_diff(list(out_ref_arr), test_out)
    if err > 1e-4:
        print(f"    FAIL: max_abs_diff={err:.2e} (threshold 1e-4)")
        return False
    print(f"    OK: max_abs_diff={err:.2e}")
    return True


# ── Per-type matmul coverage (every SpiteType) ────────────────────────────

# Byte offsets of the fp16 scale fields of one block, per type.  Same table as
# patch() in tools/verify/quant_oracle.c / quant_gpu_test.cu.  Types not listed
# here: IQ1_M (fp16 spread over four u16 nibbles), MXFP4 (E8M0 byte), NVFP4
# (four UE4M3 bytes) and the dense F32/F16/BF16 are handled in QuantMatrix.
HALF_SCALE_OFFSETS = {
    2: [0], 6: [0], 8: [0], 41: [0], 42: [0], 20: [0],            # Q4_0 Q5_0 Q8_0 Q1_0 Q2_0 IQ4_NL
    3: [0, 2], 7: [0, 2],                                          # Q4_1 Q5_1 (d, m)
    10: [80, 82], 11: [108], 12: [0, 2], 13: [0, 2], 14: [208],    # Q2_K Q3_K Q4_K Q5_K Q6_K
    16: [0], 17: [0], 22: [0], 18: [0], 21: [0], 19: [0], 23: [0], # IQ2_XXS/XS/S IQ3_XXS/S IQ1_S IQ4_XS
    34: [52], 35: [64],                                            # TQ1_0 TQ2_0
}
SPITE_TYPE_IQ1_M, SPITE_TYPE_MXFP4, SPITE_TYPE_NVFP4 = 29, 39, 40
DENSE_KINDS = (SPITE_TYPE_F32, SPITE_TYPE_F16, SPITE_TYPE_BF16)


class QuantMatrix:
    """Random packed blocks of one quant type with finite scales.

    Payload bytes are uniformly random; every scale field is then overwritten
    with a finite value (a random NaN/Inf scale would fail the tolerance for a
    reason that is not a kernel bug).  pack(mult) sets the scales to
    (random unit factor) * mult, so output magnitude is linear in mult for the
    fp16-scale types and moves in powers of two for MXFP4/NVFP4; the caller
    calibrates mult so weights have rms ~ 1/sqrt(cols) like the F32 case.
    """

    def __init__(self, kind: int, rows: int, cols: int, rng: random.Random):
        self.kind = kind
        self.bb = TYPE_BLOCK_BYTES[kind]
        be = TYPE_BLOCK_ELEMS[kind]
        assert cols % be == 0
        self.nb = rows * (cols // be)
        self.raw = rng.randbytes(self.nb * self.bb)
        self.unit = [rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.5) for _ in range(4 * self.nb)]

    def pack(self, mult: float) -> bytes:
        buf = bytearray(self.raw)
        bb, k = self.bb, self.kind
        shift = round(math.log2(mult)) if mult > 0 else 0
        for b in range(self.nb):
            base = b * bb
            if k in HALF_SCALE_OFFSETS:
                for f, off in enumerate(HALF_SCALE_OFFSETS[k]):
                    struct.pack_into("<e", buf, base + off, self.unit[4 * b + f] * mult)
            elif k == SPITE_TYPE_IQ1_M:
                h = struct.unpack("<H", struct.pack("<e", self.unit[4 * b] * mult))[0]
                sc = list(struct.unpack_from("<4H", buf, base + 48))
                for i in range(4):
                    sc[i] = (sc[i] & 0x0FFF) | (((h >> (4 * i)) & 0xF) << 12)
                struct.pack_into("<4H", buf, base + 48, *sc)
            elif k == SPITE_TYPE_MXFP4:          # E8M0: 2^(e-127); every byte but 255 is a normal scale
                buf[base] = min(254, max(2, 127 + (b % 3 - 1) + shift))
            elif k == SPITE_TYPE_NVFP4:          # UE4M3 bytes: exp field 1..14, keep random mantissa
                for i in range(4):
                    exp = min(14, max(1, 7 + ((b + i) % 3 - 1) + shift))
                    buf[base + i] = (exp << 3) | (buf[base + i] & 7)
            else:
                raise ValueError(f"no scale patch for type {k}")
        return bytes(buf)


def dense_blob(kind: int, n: int, rng: random.Random, scale: float) -> bytes:
    vals = [rng.gauss(0, scale) for _ in range(n)]
    if kind == SPITE_TYPE_F32:
        return struct.pack(f"<{n}f", *vals)
    if kind == SPITE_TYPE_F16:
        return struct.pack(f"<{n}e", *vals)
    return b"".join(struct.pack("<f", v)[2:4] for v in vals)  # BF16 = top half of the f32


def run_matmul_op(info: SpiteKernelInfo, blob: bytes, kind: int, x_data: list[float], rows: int,
                  cols: int, cuda: CudaHelper, on_gpu: bool):
    """Run info.matmul on (blob, x); returns (ret, out list or None, skip reason or None)."""
    fn = MatmulFn(info.matmul)
    tx, _ = make_tensor(x_data, [cols, 1, 1, 1])
    tw, _ = make_tensor([0.0], [cols, rows, 1, 1], kind=kind)
    to, _ = make_tensor([0.0], [rows, 1, 1, 1])
    out = (ctypes.c_float * rows)()
    ctx = make_ctx()
    if not on_gpu:
        w_buf = ctypes.create_string_buffer(blob, len(blob))
        tw.data = ctypes.cast(w_buf, ctypes.c_void_p)
        to.data = ctypes.cast(out, ctypes.c_void_p)
        ret = fn(ctypes.byref(to), ctypes.byref(tx), ctypes.byref(tw), ctypes.byref(ctx))
        return ret, list(out), None
    if not cuda.available:
        return 0, None, "CUDA runtime not available"
    d_x, d_w, d_o = cuda.malloc(cols * 4), cuda.malloc(len(blob)), cuda.malloc(rows * 4)
    try:
        cuda.h2d(d_x, (ctypes.c_float * cols)(*x_data), cols * 4)
        cuda.h2d(d_w, ctypes.create_string_buffer(blob, len(blob)), len(blob))
        tx.data, tw.data, to.data = d_x, d_w, d_o
        ret = fn(ctypes.byref(to), ctypes.byref(tx), ctypes.byref(tw), ctypes.byref(ctx))
        cuda.sync()
        if ret == 0:
            cuda.d2h(out, d_o, rows * 4)
        return ret, list(out), None
    finally:
        cuda.free(d_x)
        cuda.free(d_w)
        cuda.free(d_o)


def verify_matmul_quant(ref_info: SpiteKernelInfo, test_info: SpiteKernelInfo,
                        cuda: CudaHelper, is_cuda: bool, kind: int, rows: int, cols: int) -> str:
    """matmul of one weight type vs the generic reference; returns "ok", "skip" or "fail".

    A -1 from the kernel under test for a type it does not declare is a SKIP; a
    -1 for a declared type, any other nonzero return or a numerical mismatch
    (> 1e-4 abs, same threshold as the F32 case) is a FAIL.
    """
    name = TYPE_NAME[kind]
    print(f"\n  [matmul] type={name} shape=[{rows}, {cols}]")
    if not ref_info.matmul or not test_info.matmul:
        print("    SKIP: reference or test kernel has no matmul")
        return "skip"
    rng = random.Random(0x5EED + kind)
    x_data = [rng.gauss(0, 1) for _ in range(cols)]
    target = 1.0 / math.sqrt(cols)       # weight rms, as in verify_matmul
    if kind in DENSE_KINDS:
        blob = dense_blob(kind, rows * cols, rng, target)
    else:
        qm = QuantMatrix(kind, rows, cols, rng)
        # Calibrate the scale magnitude with the reference itself: for x ~ N(0,1)
        # rms(out) = sqrt(cols) * rms(w).
        ret, out, _ = run_matmul_op(ref_info, qm.pack(1.0), kind, x_data, rows, cols, cuda, False)
        if ret != 0 or not out:
            print(f"    SKIP: reference matmul returned {ret}")
            return "skip"
        rms_w = math.sqrt(sum(v * v for v in out) / rows / cols)
        blob = qm.pack(target / rms_w if rms_w > 0 and math.isfinite(rms_w) else 1.0)
    ret_ref, ref_out, _ = run_matmul_op(ref_info, blob, kind, x_data, rows, cols, cuda, False)
    if ret_ref != 0:
        print(f"    SKIP: reference matmul returned {ret_ref}")
        return "skip"
    ret, got, why = run_matmul_op(test_info, blob, kind, x_data, rows, cols, cuda, is_cuda)
    if why:
        print(f"    SKIP: {why}")
        return "skip"
    declared = kind == SPITE_TYPE_F32 or kind in tuple(test_info.supported_quants)
    if ret == -1 and not declared:
        print(f"    SKIP: kernel returns -1 for {name} (not in supported_quants, no library decoder)")
        return "skip"
    if ret != 0:
        print(f"    FAIL: test matmul returned {ret}" +
              (" although the type is declared in supported_quants" if ret == -1 else ""))
        return "fail"
    err = max_abs_diff(ref_out, got)
    if err > 1e-4:
        print(f"    FAIL: max_abs_diff={err:.2e} (threshold 1e-4)")
        return "fail"
    print(f"    OK: max_abs_diff={err:.2e}  (|out| max {max(abs(v) for v in ref_out):.2f})")
    return "ok"


def verify_ffn(ref_info: SpiteKernelInfo, test_info: SpiteKernelInfo,
               cuda: CudaHelper, is_cuda: bool, hidden: int = 64, ffn_dim: int = 128) -> bool:
    print(f"\n  [ffn] hidden={hidden} ffn_dim={ffn_dim}")
    if not ref_info.ffn:
        print("    SKIP: reference kernel has no ffn")
        return True
    if not test_info.ffn:
        print("    SKIP: test kernel has no ffn")
        return True

    rng = random.Random(42)
    x_data = [rng.gauss(0, 1) for _ in range(hidden)]
    s_h = 1.0 / math.sqrt(hidden)
    s_f = 1.0 / math.sqrt(ffn_dim)
    wg_data = [rng.gauss(0, s_h) for _ in range(hidden * ffn_dim)]
    wu_data = [rng.gauss(0, s_h) for _ in range(hidden * ffn_dim)]
    wd_data = [rng.gauss(0, s_f) for _ in range(hidden * ffn_dim)]
    out_init = [rng.gauss(0, 0.1) for _ in range(hidden)]

    out_ref_arr = (ctypes.c_float * hidden)(*out_init)
    tx, _ = make_tensor(x_data, [hidden, 1, 1, 1])
    twg, _ = make_tensor(wg_data, [hidden, ffn_dim, 1, 1])
    twu, _ = make_tensor(wu_data, [hidden, ffn_dim, 1, 1])
    twd, _ = make_tensor(wd_data, [ffn_dim, hidden, 1, 1])
    to, _ = make_tensor([0.0] * hidden, [hidden, 1, 1, 1])
    to.data = ctypes.cast(out_ref_arr, ctypes.c_void_p)
    ctx_ref = make_ctx()

    ref_fn = FfnFn(ref_info.ffn)
    ret_ref = ref_fn(ctypes.byref(to), ctypes.byref(tx), ctypes.byref(twg), ctypes.byref(twu),
                     ctypes.byref(twd), SPITE_FFN_SILU_GATE, ctypes.byref(ctx_ref))
    if ret_ref != 0:
        print(f"    SKIP: reference ffn returned {ret_ref}")
        return True

    test_fn = FfnFn(test_info.ffn)
    if not is_cuda:
        out_test_arr = (ctypes.c_float * hidden)(*out_init)
        to_test, _ = make_tensor([0.0] * hidden, [hidden, 1, 1, 1])
        to_test.data = ctypes.cast(out_test_arr, ctypes.c_void_p)
        ctx_test = make_ctx()
        ret_test = test_fn(ctypes.byref(to_test), ctypes.byref(tx), ctypes.byref(twg), ctypes.byref(twu),
                           ctypes.byref(twd), SPITE_FFN_SILU_GATE, ctypes.byref(ctx_test))
        if ret_test != 0:
            print(f"    FAIL: test ffn returned {ret_test}")
            return False
        test_out = list(out_test_arr)
    else:
        if not cuda.available:
            print("    SKIP: CUDA runtime not available")
            return True
        scratch_sz = ffn_dim * 2 * 4
        d_x = cuda.malloc(hidden * 4)
        d_wg = cuda.malloc(hidden * ffn_dim * 4)
        d_wu = cuda.malloc(hidden * ffn_dim * 4)
        d_wd = cuda.malloc(hidden * ffn_dim * 4)
        d_o = cuda.malloc(hidden * 4)
        d_scratch = cuda.malloc(scratch_sz)
        try:
            cuda.h2d(d_x, (ctypes.c_float * hidden)(*x_data), hidden * 4)
            cuda.h2d(d_wg, (ctypes.c_float * (hidden * ffn_dim))(*wg_data), hidden * ffn_dim * 4)
            cuda.h2d(d_wu, (ctypes.c_float * (hidden * ffn_dim))(*wu_data), hidden * ffn_dim * 4)
            cuda.h2d(d_wd, (ctypes.c_float * (hidden * ffn_dim))(*wd_data), hidden * ffn_dim * 4)
            cuda.h2d(d_o, (ctypes.c_float * hidden)(*out_init), hidden * 4)

            tx_gpu, _ = make_tensor([0.0] * hidden, [hidden, 1, 1, 1])
            twg_gpu, _ = make_tensor([0.0] * (hidden * ffn_dim), [hidden, ffn_dim, 1, 1])
            twu_gpu, _ = make_tensor([0.0] * (hidden * ffn_dim), [hidden, ffn_dim, 1, 1])
            twd_gpu, _ = make_tensor([0.0] * (hidden * ffn_dim), [ffn_dim, hidden, 1, 1])
            to_gpu, _ = make_tensor([0.0] * hidden, [hidden, 1, 1, 1])
            tx_gpu.data = d_x
            twg_gpu.data = d_wg
            twu_gpu.data = d_wu
            twd_gpu.data = d_wd
            to_gpu.data = d_o

            ctx_gpu = make_ctx()
            ctx_gpu.scratchpad = d_scratch
            ctx_gpu.scratchpad_bytes = scratch_sz

            ret_test = test_fn(ctypes.byref(to_gpu), ctypes.byref(tx_gpu), ctypes.byref(twg_gpu),
                               ctypes.byref(twu_gpu), ctypes.byref(twd_gpu), SPITE_FFN_SILU_GATE,
                               ctypes.byref(ctx_gpu))
            cuda.sync()
            if ret_test != 0:
                print(f"    FAIL: test ffn returned {ret_test}")
                return False
            out_test_arr = (ctypes.c_float * hidden)()
            cuda.d2h(out_test_arr, d_o, hidden * 4)
            test_out = list(out_test_arr)
        finally:
            cuda.free(d_x)
            cuda.free(d_wg)
            cuda.free(d_wu)
            cuda.free(d_wd)
            cuda.free(d_o)
            cuda.free(d_scratch)

    err = max_abs_diff(list(out_ref_arr), test_out)
    if err > 1e-4:
        print(f"    FAIL: max_abs_diff={err:.2e} (threshold 1e-4)")
        return False
    print(f"    OK: max_abs_diff={err:.2e}")
    return True


def verify_attention(ref_info: SpiteKernelInfo, test_info: SpiteKernelInfo,
                     cuda: CudaHelper, is_cuda: bool,
                     d_model: int = 64, head_dim: int = 16,
                     n_heads: int = 4, n_kv_heads: int = 2,
                     n_ctx: int = 8, pos: int = 3,
                     use_qk_norm: bool = True,
                     kv_kind: int = SPITE_TYPE_F32) -> bool:
    """One decode token at `pos` against a pre-populated KV cache.

    Exercises the whole op: Q/K/V projections, optional per-head QK RMSNorm,
    NEOX RoPE, the KV-cache write at row `pos`, GQA score/softmax/weighted-V
    over rows [0, pos], and the fused `out += Wo*att` residual.

    The reference is the scalar C kernel in kernels/generic/generic/ops.c.
    With kv_kind=F32 (the default) that is the tier the generic kernel declares
    (kv_cache_kinds left NULL) and the highest-fidelity path each GPU kernel
    offers, so the case pins the maths rather than the block codec.

    kv_kind selects a VBR cache tier instead.  The generic reference is F32-KV
    only, so a quantized tier needs a tier-capable --ref; both sides then read
    the same cache bytes, which makes the case a differential check of the
    kernel's block decoders (q8_0/q5_1/q4_0/f16) rather than a comparison
    against an independent implementation.

    The GPU kernels dispatch on shape: a small head_dim / short context takes
    the portable VBR back end, a supported head_dim with room for the split-K
    workspace takes the flash back end, and `pos` past one KV tile exercises
    the tiled loop and the multi-chunk combine.  Call it once per path.
    """
    tag = "" if use_qk_norm else " (no QK norm)"
    tier = KV_TIER_NAMES.get(kv_kind, str(kv_kind))
    print(f"\n  [attention] d_model={d_model} heads={n_heads} kv_heads={n_kv_heads} "
          f"head_dim={head_dim} ctx={n_ctx} pos={pos} kv={tier}{tag}")
    if not test_info.attention:
        print("    SKIP: test kernel has no attention")
        return True
    if not ref_info.attention:
        print("    SKIP: reference kernel has no attention")
        return True
    if kv_kind != SPITE_TYPE_F32 and not (declared_kv_tiers(ref_info) & (1 << kv_kind)):
        print(f"    SKIP: reference kernel does not accept {tier} KV (the generic C "
              f"reference is F32-only; pass --ref <kernel.so> to make this a "
              f"differential check of the block decoder)")
        return True

    kv_stride = n_kv_heads * head_dim
    nh_hd     = n_heads * head_dim
    eps       = 1e-5
    rope_base = 10000.0

    rng = random.Random(42)
    w_scale = 1.0 / math.sqrt(d_model)
    x_data   = [rng.gauss(0, 1.0) for _ in range(d_model)]
    wq_data  = [rng.gauss(0, w_scale) for _ in range(d_model * nh_hd)]
    wk_data  = [rng.gauss(0, w_scale) for _ in range(d_model * kv_stride)]
    wv_data  = [rng.gauss(0, w_scale) for _ in range(d_model * kv_stride)]
    wo_data  = [rng.gauss(0, w_scale) for _ in range(nh_hd * d_model)]
    qn_data  = [rng.uniform(0.5, 1.5) for _ in range(head_dim)]
    kn_data  = [rng.uniform(0.5, 1.5) for _ in range(head_dim)]
    out_init = [rng.gauss(0, 0.1) for _ in range(d_model)]
    # Rows [0, pos) are given; both sides write row `pos` themselves.  In the
    # cache's own tier, so a quantized case feeds both sides the same bytes.
    kv_blob = kv_tier_blob(kv_kind, n_ctx, kv_stride, random.Random(7))
    kv_nbytes = len(kv_blob)

    def fresh_out():
        arr = (ctypes.c_float * d_model)(*out_init)
        t, _ = make_tensor([0.0] * d_model, [d_model, 1, 1, 1])
        t.data = ctypes.cast(arr, ctypes.c_void_p)
        return t, arr

    def fresh_kv():
        ka = ctypes.create_string_buffer(kv_blob)
        va = ctypes.create_string_buffer(kv_blob)
        tk, _ = make_tensor([0.0] * (n_ctx * kv_stride), [kv_stride, n_ctx, 1, 1], kind=kv_kind)
        tv, _ = make_tensor([0.0] * (n_ctx * kv_stride), [kv_stride, n_ctx, 1, 1], kind=kv_kind)
        tk.data = ctypes.cast(ka, ctypes.c_void_p)
        tv.data = ctypes.cast(va, ctypes.c_void_p)
        kv = SpiteKvCache()
        kv.k, kv.v, kv.layer = tk, tv, 0
        return kv, ka, va

    def make_ctx_for(pos_arg):
        c = make_ctx()
        c.n_ctx      = n_ctx
        c.pos        = pos_arg
        c.n_heads    = n_heads
        c.n_kv_heads = n_kv_heads
        return c

    tx, _ = make_tensor(x_data, [d_model, 1, 1, 1])
    twq, _ = make_tensor(wq_data, [d_model, nh_hd, 1, 1])
    twk, _ = make_tensor(wk_data, [d_model, kv_stride, 1, 1])
    twv, _ = make_tensor(wv_data, [d_model, kv_stride, 1, 1])
    two, _ = make_tensor(wo_data, [nh_hd, d_model, 1, 1])
    tqn, _ = make_tensor(qn_data, [head_dim, 1, 1, 1])
    tkn, _ = make_tensor(kn_data, [head_dim, 1, 1, 1])
    p_qn = ctypes.byref(tqn) if use_qk_norm else None
    p_kn = ctypes.byref(tkn) if use_qk_norm else None

    # ── reference ──
    # A host reference (the scalar C kernel) runs in host memory; a CUDA
    # reference is a differential oracle and runs on the device tensors built
    # below, so `--ref <gpu kernel.so>` compares two GPU implementations.
    to_ref, out_ref_arr = fresh_out()
    ref_fn = AttentionFn(ref_info.attention)
    ref_is_cuda = looks_cuda(ref_info)
    if not ref_is_cuda:
        kv_ref, _, _ = fresh_kv()
        ret_ref = ref_fn(ctypes.byref(to_ref), ctypes.byref(tx), ctypes.byref(twq),
                         ctypes.byref(twk), ctypes.byref(twv), ctypes.byref(two),
                         p_qn, p_kn, eps, ctypes.byref(kv_ref), rope_base,
                         ctypes.byref(make_ctx_for(pos)))
        if ret_ref != 0:
            print(f"    FAIL: reference attention returned {ret_ref}")
            return False
    elif not is_cuda:
        print("    SKIP: reference is a CUDA kernel but the under-test kernel is not")
        return True

    test_fn = AttentionFn(test_info.attention)
    if not is_cuda:
        to_test, test_out_arr = fresh_out()
        kv_test, _, _ = fresh_kv()
        ret_test = test_fn(ctypes.byref(to_test), ctypes.byref(tx), ctypes.byref(twq),
                           ctypes.byref(twk), ctypes.byref(twv), ctypes.byref(two),
                           p_qn, p_kn, eps, ctypes.byref(kv_test), rope_base,
                           ctypes.byref(make_ctx_for(pos)))
        if ret_test != 0:
            print(f"    FAIL: test attention returned {ret_test}")
            return False
        test_out = list(test_out_arr)
    else:
        if not cuda.available:
            print("    SKIP: CUDA runtime not available")
            return True
        # Sizing mirror of kvattn_prologue's requirement, so a too-small
        # scratchpad shows up here as -2 rather than as a silent pass.
        scratch_sz = 4 * (nh_hd * 2 + kv_stride * 2 + n_heads * n_ctx)
        d_x  = cuda.malloc(d_model * 4)
        d_wq = cuda.malloc(len(wq_data) * 4)
        d_wk = cuda.malloc(len(wk_data) * 4)
        d_wv = cuda.malloc(len(wv_data) * 4)
        d_wo = cuda.malloc(len(wo_data) * 4)
        d_qn = cuda.malloc(head_dim * 4) if use_qk_norm else None
        d_kn = cuda.malloc(head_dim * 4) if use_qk_norm else None
        d_o  = cuda.malloc(d_model * 4)
        d_k  = cuda.malloc(kv_nbytes)
        d_v  = cuda.malloc(kv_nbytes)
        d_scratch = cuda.malloc(scratch_sz)
        try:
            cuda.h2d(d_x, (ctypes.c_float * d_model)(*x_data), d_model * 4)
            cuda.h2d(d_wq, (ctypes.c_float * len(wq_data))(*wq_data), len(wq_data) * 4)
            cuda.h2d(d_wk, (ctypes.c_float * len(wk_data))(*wk_data), len(wk_data) * 4)
            cuda.h2d(d_wv, (ctypes.c_float * len(wv_data))(*wv_data), len(wv_data) * 4)
            cuda.h2d(d_wo, (ctypes.c_float * len(wo_data))(*wo_data), len(wo_data) * 4)
            if use_qk_norm:
                cuda.h2d(d_qn, (ctypes.c_float * head_dim)(*qn_data), head_dim * 4)
                cuda.h2d(d_kn, (ctypes.c_float * head_dim)(*kn_data), head_dim * 4)
            cuda.h2d(d_o, (ctypes.c_float * d_model)(*out_init), d_model * 4)
            cuda.h2d(d_k, (ctypes.c_ubyte * kv_nbytes)(*kv_blob), kv_nbytes)
            cuda.h2d(d_v, (ctypes.c_ubyte * kv_nbytes)(*kv_blob), kv_nbytes)

            g_x, _  = make_tensor([0.0] * d_model, [d_model, 1, 1, 1])
            g_wq, _ = make_tensor([0.0] * len(wq_data), [d_model, nh_hd, 1, 1])
            g_wk, _ = make_tensor([0.0] * len(wk_data), [d_model, kv_stride, 1, 1])
            g_wv, _ = make_tensor([0.0] * len(wv_data), [d_model, kv_stride, 1, 1])
            g_wo, _ = make_tensor([0.0] * len(wo_data), [nh_hd, d_model, 1, 1])
            g_qn, _ = make_tensor([0.0] * head_dim, [head_dim, 1, 1, 1])
            g_kn, _ = make_tensor([0.0] * head_dim, [head_dim, 1, 1, 1])
            g_o, _  = make_tensor([0.0] * d_model, [d_model, 1, 1, 1])
            g_k, _  = make_tensor([0.0] * (n_ctx * kv_stride), [kv_stride, n_ctx, 1, 1], kind=kv_kind)
            g_v, _  = make_tensor([0.0] * (n_ctx * kv_stride), [kv_stride, n_ctx, 1, 1], kind=kv_kind)
            g_x.data, g_wq.data, g_wk.data, g_wv.data = d_x, d_wq, d_wk, d_wv
            g_wo.data, g_o.data, g_k.data, g_v.data = d_wo, d_o, d_k, d_v
            if use_qk_norm:
                g_qn.data, g_kn.data = d_qn, d_kn

            g_kv = SpiteKvCache()
            g_kv.k, g_kv.v, g_kv.layer = g_k, g_v, 0
            ctx_gpu = make_ctx_for(pos)
            ctx_gpu.scratchpad = d_scratch
            ctx_gpu.scratchpad_bytes = scratch_sz

            if ref_is_cuda:
                ret_ref = ref_fn(ctypes.byref(g_o), ctypes.byref(g_x), ctypes.byref(g_wq),
                                 ctypes.byref(g_wk), ctypes.byref(g_wv), ctypes.byref(g_wo),
                                 ctypes.byref(g_qn) if use_qk_norm else None,
                                 ctypes.byref(g_kn) if use_qk_norm else None,
                                 eps, ctypes.byref(g_kv), rope_base, ctypes.byref(ctx_gpu))
                cuda.sync()
                if ret_ref != 0:
                    print(f"    FAIL: reference attention returned {ret_ref}")
                    return False
                cuda.d2h(out_ref_arr, d_o, d_model * 4)
                # both sides accumulate into `out`, so restore its input, and
                # the KV row `pos` they rewrite holds the same values either way
                cuda.h2d(d_o, (ctypes.c_float * d_model)(*out_init), d_model * 4)

            ret_test = test_fn(ctypes.byref(g_o), ctypes.byref(g_x), ctypes.byref(g_wq),
                               ctypes.byref(g_wk), ctypes.byref(g_wv), ctypes.byref(g_wo),
                               ctypes.byref(g_qn) if use_qk_norm else None,
                               ctypes.byref(g_kn) if use_qk_norm else None,
                               eps, ctypes.byref(g_kv), rope_base, ctypes.byref(ctx_gpu))
            cuda.sync()
            if ret_test != 0:
                print(f"    FAIL: test attention returned {ret_test}")
                return False
            test_out_arr = (ctypes.c_float * d_model)()
            cuda.d2h(test_out_arr, d_o, d_model * 4)
            test_out = list(test_out_arr)
        finally:
            for p in (d_x, d_wq, d_wk, d_wv, d_wo, d_qn, d_kn, d_o, d_k, d_v, d_scratch):
                if p:
                    cuda.free(p)

    err = max_abs_diff(list(out_ref_arr), test_out)
    if err > 1e-4:
        print(f"    FAIL: max_abs_diff={err:.2e} (threshold 1e-4)")
        return False
    print(f"    OK: max_abs_diff={err:.2e}")
    return True


# ── ABI v7 layer ops: Gated Delta Net layer + extended attention ──────────

# Which SpiteType each projection uses.  The K-quant row uses one of each of
# Q4_K / Q5_K / Q6_K, with Q8_0 on the small projections, like the shipped GGUF.
GDN_WEIGHT_SETS = {
    "F32":  dict(qkv=SPITE_TYPE_F32, gate=SPITE_TYPE_F32, beta=SPITE_TYPE_F32,
                 alpha=SPITE_TYPE_F32, out=SPITE_TYPE_F32),
    "Q8_0": dict(qkv=SPITE_TYPE_Q8_0, gate=SPITE_TYPE_Q8_0, beta=SPITE_TYPE_Q8_0,
                 alpha=SPITE_TYPE_Q8_0, out=SPITE_TYPE_Q8_0),
    "Q4_K/Q5_K/Q6_K": dict(qkv=SPITE_TYPE_Q5_K, gate=SPITE_TYPE_Q4_K, beta=SPITE_TYPE_Q8_0,
                           alpha=SPITE_TYPE_Q8_0, out=SPITE_TYPE_Q6_K),
}
ATTN_WEIGHT_SETS = {
    "F32":  dict(q=SPITE_TYPE_F32, k=SPITE_TYPE_F32, v=SPITE_TYPE_F32, o=SPITE_TYPE_F32),
    "Q8_0": dict(q=SPITE_TYPE_Q8_0, k=SPITE_TYPE_Q8_0, v=SPITE_TYPE_Q8_0, o=SPITE_TYPE_Q8_0),
    "Q4_K/Q5_K/Q6_K": dict(q=SPITE_TYPE_Q4_K, k=SPITE_TYPE_Q5_K, v=SPITE_TYPE_Q6_K,
                           o=SPITE_TYPE_Q8_0),
}
# Layer outputs of packed weights are compared at the Q8-class tolerance; the
# two sides read identical bytes, only accumulation order differs.
TOL_F32, TOL_PACKED = 1e-4, 1e-3


def f32b(vals) -> bytes:
    return array("f", vals).tobytes()


def unf32(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


class Buf:
    """Bytes on the host or the CUDA device, wrapped as one contiguous SpiteTensor."""

    def __init__(self, cuda: CudaHelper, on_gpu: bool, blob: bytes,
                 kind: int = SPITE_TYPE_F32, ne=(1, 1, 1, 1)):
        self.cuda, self.on_gpu, self.n = cuda, on_gpu, len(blob)
        self.host = ctypes.create_string_buffer(bytes(blob), max(self.n, 1))
        self.dev = cuda.malloc(max(self.n, 1)) if on_gpu else None
        self.ptr = self.dev if on_gpu else ctypes.addressof(self.host)
        if on_gpu and self.n:
            cuda.h2d(self.dev, self.host, self.n)
        self.t = SpiteTensor()
        self.t.data = self.ptr
        for i in range(4):
            self.t.ne[i] = ne[i]
        for i, v in enumerate(contiguous_strides(kind, list(ne))):
            self.t.nb[i] = v
        self.t.kind = kind

    def set(self, blob: bytes):
        assert len(blob) == self.n
        ctypes.memmove(self.host, blob, self.n)
        if self.on_gpu and self.n:
            self.cuda.h2d(self.dev, self.host, self.n)

    def get(self) -> bytes:
        if self.on_gpu and self.n:
            self.cuda.d2h(self.host, self.dev, self.n)
        return self.host.raw[:self.n]

    def free(self):
        if self.on_gpu:
            self.cuda.free(self.dev)


def make_scratch(cuda: CudaHelper, on_gpu: bool, mode: str, nbytes: int, ctx: SpiteCtx):
    """Attach a scratchpad to ctx.  mode: "none" (host only), "exact", or "short" (one float
    less than needed).  The scratch is filled with NaN bytes: an op that reads scratch before
    writing it fails loudly, as it would when the host reuses one buffer across layers."""
    if mode == "none":
        assert not on_gpu
        return None
    size = nbytes - 4 if mode == "short" else nbytes
    sb = Buf(cuda, on_gpu, b"\xff" * size, SPITE_TYPE_F32, (max(size // 4, 1), 1, 1, 1))
    ctx.scratchpad = sb.ptr
    ctx.scratchpad_bytes = size
    return sb


_RMS1 = {}    # kind -> rms of the weights QuantMatrix.pack(1.0) decodes to


def q8_0_blob(rows: int, cols: int, rng: random.Random, rms: float) -> bytes:
    """Random Q8_0 [rows x cols] weights of the given rms, built without a per-block
    python loop (the 27B-shape projections are 50 MB): uniform int8 payload (rms 73.9),
    fp16 scales taken from a 1021-entry pattern."""
    assert cols % 32 == 0
    nb = rows * cols // 32
    raw = bytearray(rng.randbytes(nb * 34))
    pat = [struct.pack("<e", rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.5) * rms / 73.9)
           for _ in range(1021)]
    reps = nb // 1021 + 1
    raw[0::34] = (bytes(p[0] for p in pat) * reps)[:nb]
    raw[1::34] = (bytes(p[1] for p in pat) * reps)[:nb]
    return bytes(raw)


def weight_blob(kind: int, rows: int, cols: int, rng: random.Random,
                ref_info: SpiteKernelInfo, cuda: CudaHelper) -> bytes:
    """Random [cols, rows] weights of `kind` with rms ~ 1/sqrt(cols), as in the F32 cases."""
    target = 1.0 / math.sqrt(cols)
    if kind == SPITE_TYPE_F32:
        return f32b(rng.gauss(0, target) for _ in range(rows * cols))
    if kind == SPITE_TYPE_Q8_0:
        return q8_0_blob(rows, cols, rng, target)
    qm = QuantMatrix(kind, rows, cols, rng)
    if kind not in _RMS1:
        # calibrate with the reference matmul, as verify_matmul_quant does
        cal_rows, cal_cols = 64, 1024
        cal = QuantMatrix(kind, cal_rows, cal_cols, random.Random(0xCA1 + kind))
        xc = [random.Random(kind).gauss(0, 1) for _ in range(cal_cols)]
        ret, out, _ = run_matmul_op(ref_info, cal.pack(1.0), kind, xc, cal_rows, cal_cols, cuda, False)
        rms = math.sqrt(sum(v * v for v in out) / cal_rows / cal_cols) if ret == 0 else 0.0
        _RMS1[kind] = rms if rms > 0 and math.isfinite(rms) else 1.0
    return qm.pack(target / _RMS1[kind])


def decode_weight(ref_info: SpiteKernelInfo, kind: int, blob: bytes, n: int):
    """Decode a packed weight to floats with the reference library's spite_dequantize_row
    (bit-exact vs ggml per tools/verify/quant_oracle.c); None if unavailable."""
    if kind == SPITE_TYPE_F32:
        return unf32(blob)
    lib = getattr(ref_info, "_lib", None)
    fn = getattr(lib, "spite_dequantize_row", None) if lib is not None else None
    if fn is None:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    out = (ctypes.c_float * n)()
    buf = ctypes.create_string_buffer(blob, len(blob))
    return list(out) if fn(kind, ctypes.addressof(buf), out, n) == 0 else None


def matvec(w: list[float], rows: int, x: list[float]) -> list[float]:
    cols = len(x)
    return [sum(map(operator.mul, w[r * cols:(r + 1) * cols], x)) for r in range(rows)]


def kind_names(kinds: dict) -> str:
    return ",".join(sorted({TYPE_NAME[k] for k in kinds.values()}))


# ── Gated Delta Net layer ─────────────────────────────────────────────────

class GdnCase:
    """Random inputs for `n_steps` sequential decode tokens of one GDN geometry."""

    def __init__(self, ref_info, cuda, n_kh, n_vh, S, K, d_model, kinds, n_steps, seed=7):
        rng = random.Random(seed)
        self.n_kh, self.n_vh, self.S, self.K, self.d_model = n_kh, n_vh, S, K, d_model
        self.kd, self.vd = n_kh * S, n_vh * S
        self.C = 2 * self.kd + self.vd
        self.eps, self.n_steps, self.kinds = 1e-6, n_steps, kinds
        self.rows = dict(qkv=self.C, gate=self.vd, beta=n_vh, alpha=n_vh, out=d_model)
        self.cols = dict(qkv=d_model, gate=d_model, beta=d_model, alpha=d_model, out=self.vd)
        self.w = {n: weight_blob(kinds[n], self.rows[n], self.cols[n], rng, ref_info, cuda)
                  for n in ("qkv", "gate", "beta", "alpha", "out")}
        self.conv_w = [rng.uniform(-0.6, 0.6) for _ in range(K * self.C)]   # tap k of channel c at c*K+k
        self.dt = [rng.uniform(-0.5, 0.5) for _ in range(n_vh)]
        self.a = [-rng.uniform(0.1, 2.0) for _ in range(n_vh)]
        self.nrm = [rng.uniform(0.5, 1.5) for _ in range(S)]
        self.xs = [[rng.gauss(0, 1) for _ in range(d_model)] for _ in range(n_steps)]
        self.out_init = [[rng.gauss(0, 0.1) for _ in range(d_model)] for _ in range(n_steps)]

    def scratch_bytes(self) -> int:           # spite_gdn_scratch_floats
        return 4 * (2 * self.kd + 3 * self.vd + 2 * self.n_vh)

    def describe(self) -> str:
        return (f"n_kh={self.n_kh} n_vh={self.n_vh} S={self.S} K={self.K} d_model={self.d_model} "
                f"weights={kind_names(self.kinds)} steps={self.n_steps}")


GDN_ARGS = ("out", "x", "w_qkv", "w_gate", "w_beta", "w_alpha", "w_out", "conv_w", "ssm_dt",
            "ssm_a", "ssm_norm", "conv_hist", "state")


def run_gdn(info, on_gpu, cuda, case: GdnCase, scratch="exact", tamper=None):
    """Run the case's tokens on one kernel's linear_attn.  Returns (ret, snaps): ret is the
    first nonzero return code (0 if none), snaps[i] = (out, state, conv_hist) as floats after
    call i (the failing call included, to show an error leaves everything untouched).
    tamper(bufs, params, ctx) may corrupt the call (contract tests)."""
    c = case
    z = lambda n: b"\x00" * (4 * n)
    mk = lambda blob, kind, ne: Buf(cuda, on_gpu, blob, kind, ne)
    bufs = {
        "out": mk(f32b(c.out_init[0]), SPITE_TYPE_F32, (c.d_model, 1, 1, 1)),
        "x": mk(f32b(c.xs[0]), SPITE_TYPE_F32, (c.d_model, 1, 1, 1)),
        "conv_w": mk(f32b(c.conv_w), SPITE_TYPE_F32, (c.K, c.C, 1, 1)),
        "ssm_dt": mk(f32b(c.dt), SPITE_TYPE_F32, (c.n_vh, 1, 1, 1)),
        "ssm_a": mk(f32b(c.a), SPITE_TYPE_F32, (c.n_vh, 1, 1, 1)),
        "ssm_norm": mk(f32b(c.nrm), SPITE_TYPE_F32, (c.S, 1, 1, 1)),
        "conv_hist": mk(z((c.K - 1) * c.C), SPITE_TYPE_F32, (c.C, c.K - 1, 1, 1)),
        "state": mk(z(c.n_vh * c.S * c.S), SPITE_TYPE_F32, (c.S, c.S, c.n_vh, 1)),
    }
    for n in ("qkv", "gate", "beta", "alpha", "out"):
        bufs["w_" + n] = mk(c.w[n], c.kinds[n], (c.cols[n], c.rows[n], 1, 1))
    own = dict(bufs)
    params = SpiteGdnParams(c.n_kh, c.n_vh, c.S, c.K, c.eps)
    ctx = make_ctx()
    scr = make_scratch(cuda, on_gpu, "exact" if on_gpu else scratch, c.scratch_bytes(), ctx)
    try:
        if tamper:
            tamper(bufs, params, ctx)
        fn = GdnFn(info.linear_attn)
        ptrs = [ctypes.byref(bufs[n].t) if bufs[n] is not None else None for n in GDN_ARGS]
        snaps = []
        for i in range(c.n_steps):
            own["x"].set(f32b(c.xs[i]))
            own["out"].set(f32b(c.out_init[i]))
            ret = fn(*ptrs, ctypes.byref(params), ctypes.byref(ctx))
            if on_gpu:
                cuda.sync()
            snaps.append(tuple(unf32(own[n].get()) for n in ("out", "state", "conv_hist")))
            if ret != 0:
                return ret, snaps
        return 0, snaps
    finally:
        for b in own.values():
            b.free()
        if scr:
            scr.free()


def gdn_oracle(c: GdnCase, W: dict):
    """Independent float64 model of SpiteGdnFn written from the ABI semantics (plain
    python lists, nothing shared with the C code).  W maps qkv/gate/beta/alpha/out to
    decoded row-major weights.  Returns [(out, state, conv_hist)] per step."""
    n_kh, n_vh, S, K, kd, vd, C = c.n_kh, c.n_vh, c.S, c.K, c.kd, c.vd, c.C
    hist = [[0.0] * C for _ in range(K - 1)]
    M = [[[0.0] * S for _ in range(S)] for _ in range(n_vh)]
    silu = lambda v: v / (1.0 + math.exp(-v))
    res = []
    for t in range(c.n_steps):
        x = c.xs[t]
        qkv = matvec(W["qkv"], C, x)
        z = matvec(W["gate"], vd, x)
        beta_raw = matvec(W["beta"], n_vh, x)
        alpha = matvec(W["alpha"], n_vh, x)
        conv = []
        for ch in range(C):                                   # window = [hist (oldest first) | input]
            window = [hist[i][ch] for i in range(K - 1)] + [qkv[ch]]
            conv.append(silu(sum(window[k] * c.conv_w[ch * K + k] for k in range(K))))
        if K > 1:
            hist = hist[1:] + [list(qkv)]

        def l2(v):
            ms = sum(e * e for e in v) / S
            return [e / math.sqrt(ms + c.eps / S) / math.sqrt(S) for e in v]
        q = [l2(conv[h * S:(h + 1) * S]) for h in range(n_kh)]
        k = [l2(conv[kd + h * S:kd + (h + 1) * S]) for h in range(n_kh)]
        y = []
        for vh in range(n_vh):
            kh = vh % n_kh
            v = conv[2 * kd + vh * S:2 * kd + (vh + 1) * S]
            beta = 1.0 / (1.0 + math.exp(-beta_raw[vh]))
            zz = alpha[vh] + c.dt[vh]
            g = (zz if zz > 20 else math.log1p(math.exp(zz))) * c.a[vh]
            m, kk = M[vh], k[kh]
            qq = [e / math.sqrt(S) for e in q[kh]]
            decay = math.exp(g)
            for r in range(S):
                row = m[r]
                for s_ in range(S):
                    row[s_] *= decay
            d = [(v[s_] - sum(m[r][s_] * kk[r] for r in range(S))) * beta for s_ in range(S)]
            for r in range(S):
                row = m[r]
                for s_ in range(S):
                    row[s_] += kk[r] * d[s_]
            o = [sum(m[r][s_] * qq[r] for r in range(S)) for s_ in range(S)]
            ms = sum(e * e for e in o) / S
            y += [o[s_] / math.sqrt(ms + c.eps) * c.nrm[s_] * silu(z[vh * S + s_]) for s_ in range(S)]
        proj = matvec(W["out"], c.d_model, y)
        out = [c.out_init[t][r] + proj[r] for r in range(c.d_model)]
        res.append((out, [e for mm in M for row in mm for e in row], [e for row in hist for e in row]))
    return res


def _compare(label_names, ref, got, tol, skip=()):
    """Worst |ref - got| over the named per-step arrays; returns (worst, failure message or None)."""
    worst = 0.0
    for i, (r, g) in enumerate(zip(ref, got)):
        for name, rv, gv in zip(label_names, r, g):
            if name in skip:
                continue
            e = max_abs_diff(rv, gv)
            worst = max(worst, e)
            if e > tol:
                return worst, f"{name} step {i} max_abs_diff={e:.2e} (threshold {tol:.0e})"
    return worst, None


def verify_linear_attn(ref_info, test_info, cuda, is_cuda, n_kh=2, n_vh=4, S=64, K=4,
                       d_model=256, wset="F32", n_steps=3, oracle=True) -> bool:
    """GDN layer (out += W_out . gated_norm(core(W x))) over sequential tokens: the reference
    is checked against the float64 oracle, the kernel under test against the reference."""
    kinds = GDN_WEIGHT_SETS[wset]
    print(f"\n  [linear_attn] n_kh={n_kh} n_vh={n_vh} S={S} K={K} d_model={d_model} "
          f"weights={wset} steps={n_steps}")
    if not ref_info.linear_attn:
        print("    SKIP: reference kernel has no linear_attn")
        return True
    if not test_info.linear_attn:
        print("    SKIP: test kernel has no linear_attn")
        return True
    ref_gpu = looks_cuda(ref_info)
    if ref_gpu and not is_cuda:
        print("    SKIP: reference is a CUDA kernel but the under-test kernel is not")
        return True
    if (ref_gpu or is_cuda) and not cuda.available:
        print("    SKIP: CUDA runtime not available")
        return True
    packed = any(k != SPITE_TYPE_F32 for k in kinds.values())
    tol = TOL_PACKED if packed else TOL_F32

    case = GdnCase(ref_info, cuda, n_kh, n_vh, S, K, d_model, kinds, n_steps)
    ret, ref = run_gdn(ref_info, ref_gpu, cuda, case, "none")
    if ret != 0:
        print(f"    FAIL: reference linear_attn returned {ret}")
        return False

    note = ""
    if oracle and ref_info.model_arch == b"generic":
        W = {n: decode_weight(ref_info, kinds[n], case.w[n], case.rows[n] * case.cols[n])
             for n in case.w}
        if any(v is None for v in W.values()):
            note = " (oracle skipped: reference library does not export spite_dequantize_row)"
        else:
            worst, msg = _compare(("out", "state", "conv_hist"), ref, gdn_oracle(case, W), TOL_F32)
            if msg:
                print(f"    FAIL: reference deviates from the float64 oracle: {msg}")
                return False
            note = f"; reference vs float64 oracle {worst:.2e}"
    if not is_cuda and test_info.linear_attn == ref_info.linear_attn:
        print(f"    OK: under-test is the reference{note or ' (no oracle at this size)'}")
        return True

    ret, got = run_gdn(test_info, is_cuda, cuda, case, "exact")
    if ret == -1:
        print("    SKIP: test linear_attn returned -1 (geometry or weight type unsupported)")
        return True
    if ret != 0:
        print(f"    FAIL: test linear_attn returned {ret}")
        return False
    # conv_hist storage is private to a kernel (ABI), so only out and state are compared.
    worst, msg = _compare(("out", "state", "conv_hist"), ref, got, tol, skip=("conv_hist",))
    if msg:
        print(f"    FAIL: {msg}")
        return False
    print(f"    OK: max_abs_diff={worst:.2e} (threshold {tol:.0e}){note}")
    return True


# ── Extended attention (partial RoPE + gated Q) ──────────────────────────

class AttnCase:
    """Random inputs for `n_tok` sequential tokens starting at row `pos0` of a KV cache whose
    rows [0, pos0) are pre-populated."""

    def __init__(self, ref_info, cuda, d_model, n_heads, n_kv, hd, rope_dim, gated, n_ctx,
                 pos0, n_tok, kinds, use_qk_norm=True, kv_kind=SPITE_TYPE_F32, seed=11):
        rng = random.Random(seed)
        self.d_model, self.nh, self.nkv, self.hd, self.rd = d_model, n_heads, n_kv, hd, rope_dim
        self.gated, self.n_ctx, self.pos0, self.n_tok = gated, n_ctx, pos0, n_tok
        self.kinds, self.kv_kind, self.eps, self.base = kinds, kv_kind, 1e-6, 10000.0
        self.kv_stride, self.nhd = n_kv * hd, n_heads * hd
        self.q_stride = 2 * hd if gated else hd
        self.rows = dict(q=n_heads * self.q_stride, k=self.kv_stride, v=self.kv_stride, o=d_model)
        self.cols = dict(q=d_model, k=d_model, v=d_model, o=self.nhd)
        self.w = {n: weight_blob(kinds[n], self.rows[n], self.cols[n], rng, ref_info, cuda)
                  for n in ("q", "k", "v", "o")}
        self.qn = [rng.uniform(0.5, 1.5) for _ in range(hd)] if use_qk_norm else None
        self.kn = [rng.uniform(0.5, 1.5) for _ in range(hd)] if use_qk_norm else None
        self.xs = [[rng.gauss(0, 1) for _ in range(d_model)] for _ in range(n_tok)]
        self.out_init = [[rng.gauss(0, 0.1) for _ in range(d_model)] for _ in range(n_tok)]
        # Rows [0, pos0) are given; the kernel writes the rest.  K and V differ.
        self.kv_k = kv_tier_blob(kv_kind, n_ctx, self.kv_stride, random.Random(seed + 1))
        self.kv_v = kv_tier_blob(kv_kind, n_ctx, self.kv_stride, random.Random(seed + 2))

    @property
    def plain(self) -> bool:                  # reproduces the ABI v4 `attention` op
        return not self.gated and self.rd == self.hd

    def describe(self, wset: str) -> str:
        return (f"d_model={self.d_model} heads={self.nh} kv_heads={self.nkv} head_dim={self.hd} "
                f"rope_dim={self.rd} gated={self.gated} ctx={self.n_ctx} pos={self.pos0}+{self.n_tok} "
                f"weights={wset} kv={KV_TIER_NAMES.get(self.kv_kind)}"
                + ("" if self.qn else " (no QK norm)"))


def run_attn(info, op, on_gpu, cuda, c: AttnCase, scratch="exact", tamper=None, scratch_floats=None):
    """Run the case's tokens through info.attention ("attention") or info.attention_ex ("ex").
    Returns (ret, outs, kv): first nonzero return code, the out vector after every call, and
    the K/V cache contents as floats (F32 tier only, else None).
    tamper(bufs, params, ctx) may corrupt the call; scratch_floats overrides the ABI's scratch
    size (contract tests)."""
    mk = lambda blob, kind, ne: Buf(cuda, on_gpu, blob, kind, ne)
    f32 = SPITE_TYPE_F32
    bufs = {
        "out": mk(f32b(c.out_init[0]), f32, (c.d_model, 1, 1, 1)),
        "x": mk(f32b(c.xs[0]), f32, (c.d_model, 1, 1, 1)),
        "wq": mk(c.w["q"], c.kinds["q"], (c.d_model, c.rows["q"], 1, 1)),
        "wk": mk(c.w["k"], c.kinds["k"], (c.d_model, c.rows["k"], 1, 1)),
        "wv": mk(c.w["v"], c.kinds["v"], (c.d_model, c.rows["v"], 1, 1)),
        "wo": mk(c.w["o"], c.kinds["o"], (c.nhd, c.d_model, 1, 1)),
        "q_norm": mk(f32b(c.qn), f32, (c.hd, 1, 1, 1)) if c.qn else None,
        "k_norm": mk(f32b(c.kn), f32, (c.hd, 1, 1, 1)) if c.kn else None,
        "k": mk(c.kv_k, c.kv_kind, (c.kv_stride, c.n_ctx, 1, 1)),
        "v": mk(c.kv_v, c.kv_kind, (c.kv_stride, c.n_ctx, 1, 1)),
    }
    own = {n: b for n, b in bufs.items() if b is not None}
    params = SpiteAttnParams(c.hd, c.rd, c.gated)
    ctx = make_ctx()
    ctx.n_ctx, ctx.n_heads, ctx.n_kv_heads = c.n_ctx, c.nh, c.nkv
    if op == "ex":      # spite_attn_ex_scratch_floats
        sfl = 3 * c.nh * c.hd + 2 * c.nkv * c.hd + c.nh * c.n_ctx
    else:               # the v4 op's sizing (see verify_attention)
        sfl = 2 * c.nh * c.hd + 2 * c.nkv * c.hd + c.nh * c.n_ctx
    scr = make_scratch(cuda, on_gpu, "exact" if on_gpu else scratch, 4 * (scratch_floats or sfl), ctx)
    ctx.pos = c.pos0
    try:
        if tamper:
            tamper(bufs, params, ctx)
        pos_fixed = ctx.pos != c.pos0          # a contract test pinned ctx.pos itself
        kv = SpiteKvCache()
        kv.k, kv.v, kv.layer = bufs["k"].t, bufs["v"].t, 0
        ptrs = [ctypes.byref(bufs[n].t) if bufs[n] is not None else None
                for n in ("out", "x", "wq", "wk", "wv", "wo", "q_norm", "k_norm")]
        fn = AttentionExFn(info.attention_ex) if op == "ex" else AttentionFn(info.attention)
        outs = []
        for t in range(c.n_tok):
            own["x"].set(f32b(c.xs[t]))
            own["out"].set(f32b(c.out_init[t]))
            if not pos_fixed:
                ctx.pos = c.pos0 + t
            tail = (ctypes.byref(params), ctypes.byref(ctx)) if op == "ex" else (ctypes.byref(ctx),)
            ret = fn(*ptrs, c.eps, ctypes.byref(kv), c.base, *tail)
            if on_gpu:
                cuda.sync()
            outs.append(unf32(own["out"].get()))
            if ret != 0:
                return ret, outs, None
        kvf = (unf32(own["k"].get()), unf32(own["v"].get())) if c.kv_kind == f32 else None
        return 0, outs, kvf
    finally:
        for b in own.values():
            b.free()
        if scr:
            scr.free()


def attn_snaps(c: AttnCase, outs, kvf):
    """Per-token (out, K row, V row) from a run; rows are None when the tier is not F32."""
    res = []
    for t in range(c.n_tok):
        row = c.pos0 + t
        if kvf is None:
            res.append((outs[t], [], []))
        else:
            res.append((outs[t], kvf[0][row * c.kv_stride:(row + 1) * c.kv_stride],
                        kvf[1][row * c.kv_stride:(row + 1) * c.kv_stride]))
    return res


def attn_ex_oracle(c: AttnCase, W: dict):
    """Independent float64 model of SpiteAttentionExFn (F32 KV), from the ABI semantics.
    Returns [(out, K row, V row)] per token."""
    nh, nkv, hd, rd, kvs, nhd = c.nh, c.nkv, c.hd, c.rd, c.kv_stride, c.nhd
    group, qs = nh // nkv, c.q_stride
    Kc = [list(r) for r in (unf32(c.kv_k)[i * kvs:(i + 1) * kvs] for i in range(c.n_ctx))]
    Vc = [list(r) for r in (unf32(c.kv_v)[i * kvs:(i + 1) * kvs] for i in range(c.n_ctx))]
    res = []
    for t in range(c.n_tok):
        pos, x = c.pos0 + t, c.xs[t]
        qf, kf, vf = matvec(W["q"], nh * qs, x), matvec(W["k"], kvs, x), matvec(W["v"], kvs, x)

        def norm_rope(v, w):
            if w is not None:
                r = 1.0 / math.sqrt(sum(e * e for e in v) / hd + c.eps)
                v = [e * r * wi for e, wi in zip(v, w)]
            v = list(v)
            for i in range(rd // 2):                    # NEOX pairs (i, i + rd/2) of the first rd dims
                ang = pos * c.base ** (-2.0 * i / rd)
                cs, sn = math.cos(ang), math.sin(ang)
                a, b = v[i], v[i + rd // 2]
                v[i], v[i + rd // 2] = a * cs - b * sn, a * sn + b * cs
            return v
        q = [norm_rope(qf[h * qs:h * qs + hd], c.qn) for h in range(nh)]
        gate = [qf[h * qs + hd:h * qs + 2 * hd] for h in range(nh)] if c.gated else None
        krow = [e for h in range(nkv) for e in norm_rope(kf[h * hd:(h + 1) * hd], c.kn)]
        Kc[pos], Vc[pos] = krow, vf
        att = []
        for h in range(nh):
            g = h // group
            sc = [sum(map(operator.mul, q[h], Kc[tt][g * hd:(g + 1) * hd])) / math.sqrt(hd)
                  for tt in range(pos + 1)]
            m = max(sc)
            e = [math.exp(v - m) for v in sc]
            z = sum(e)
            o = [sum(e[tt] * Vc[tt][g * hd + i] for tt in range(pos + 1)) / z for i in range(hd)]
            if c.gated:
                o = [oi / (1.0 + math.exp(-gi)) for oi, gi in zip(o, gate[h])]
            att += o
        proj = matvec(W["o"], c.d_model, att)
        res.append(([c.out_init[t][r] + proj[r] for r in range(c.d_model)], krow, vf))
    return res


def verify_attention_ex(ref_info, test_info, cuda, is_cuda, d_model=256, n_heads=4, n_kv_heads=2,
                        head_dim=64, rope_dim=32, gated=1, n_ctx=48, pos=17, n_tok=3,
                        wset="F32", use_qk_norm=True, kv_kind=SPITE_TYPE_F32,
                        oracle=True) -> bool:
    """attention_ex over `n_tok` sequential tokens (the KV cache carries between calls, pos
    advances).  The reference is checked against the float64 oracle; the kernel under test
    against the reference (output and the K/V rows it wrote).  A plain case (gated=0,
    rope_dim=head_dim) must also reproduce the ABI v4 `attention` op."""
    kinds = ATTN_WEIGHT_SETS[wset]
    tier = KV_TIER_NAMES.get(kv_kind, str(kv_kind))
    print(f"\n  [attention_ex] d_model={d_model} heads={n_heads} kv_heads={n_kv_heads} "
          f"head_dim={head_dim} rope_dim={rope_dim} gated={gated} ctx={n_ctx} pos={pos}+{n_tok} "
          f"weights={wset} kv={tier}" + ("" if use_qk_norm else " (no QK norm)"))
    if not test_info.attention_ex:
        print("    SKIP: test kernel has no attention_ex")
        return True
    if not ref_info.attention_ex:
        print("    SKIP: reference kernel has no attention_ex")
        return True
    if kv_kind != SPITE_TYPE_F32 and not (declared_kv_tiers(ref_info) & (1 << kv_kind)):
        print(f"    SKIP: reference kernel does not accept {tier} KV (pass --ref <tier-capable "
              f"kernel.so> to make this a differential check of the block decoder)")
        return True
    ref_gpu = looks_cuda(ref_info)
    if ref_gpu and not is_cuda:
        print("    SKIP: reference is a CUDA kernel but the under-test kernel is not")
        return True
    if (ref_gpu or is_cuda) and not cuda.available:
        print("    SKIP: CUDA runtime not available")
        return True
    packed = any(k != SPITE_TYPE_F32 for k in kinds.values())
    tol = TOL_PACKED if packed else TOL_F32

    c = AttnCase(ref_info, cuda, d_model, n_heads, n_kv_heads, head_dim, rope_dim, gated, n_ctx,
                 pos, n_tok, kinds, use_qk_norm, kv_kind)
    ret, outs, kvf = run_attn(ref_info, "ex", ref_gpu, cuda, c, "none")
    if ret != 0:
        print(f"    FAIL: reference attention_ex returned {ret}")
        return False
    ref = attn_snaps(c, outs, kvf)
    names = ("out", "K row", "V row")

    note = ""
    if oracle and ref_info.model_arch == b"generic" and kv_kind == SPITE_TYPE_F32:
        W = {n: decode_weight(ref_info, kinds[n], c.w[n], c.rows[n] * c.cols[n])
             for n in c.w}
        if any(v is None for v in W.values()):
            note = " (oracle skipped: reference library does not export spite_dequantize_row)"
        else:
            worst, msg = _compare(names, ref, attn_ex_oracle(c, W), TOL_F32)
            if msg:
                print(f"    FAIL: reference deviates from the float64 oracle: {msg}")
                return False
            note = f"; reference vs float64 oracle {worst:.2e}"

    if c.plain:
        # attention_ex with gated_q=0, rope_dim=head_dim IS the v4 op on every kernel that has
        # both.  The reference is bit-exact by construction (attention is a wrapper); a GPU
        # kernel's v4 op is only exercised with the F32 weights its own verify case uses.
        checks = [("reference", ref_info, ref_gpu, ref, ref_info.model_arch == b"generic")]
        if wset == "F32" and not (not is_cuda and test_info.attention_ex == ref_info.attention_ex):
            checks.append(("test kernel", test_info, is_cuda, None, False))
        for label, info, gpu, r_ex, exact in checks:
            if not info.attention:
                continue
            if r_ex is None:
                rr, o_ex, k_ex = run_attn(info, "ex", gpu, cuda, c, "exact")
                if rr != 0:
                    continue          # reported by the main comparison below
                r_ex = attn_snaps(c, o_ex, k_ex)
            r4, o4, k4 = run_attn(info, "attention", gpu, cuda, c, "exact" if gpu else "none")
            if r4 != 0:
                print(f"    SKIP plain-vs-v4 check on the {label}: attention returned {r4}")
                continue
            worst, msg = _compare(names, r_ex, attn_snaps(c, o4, k4), 0.0 if exact else tol)
            if msg:
                print(f"    FAIL: {label} attention_ex(gated=0, rope_dim=head_dim) != attention: {msg}")
                return False
            note += f"; {label} == v4 attention ({'bit-exact' if exact else f'{worst:.2e}'})"

    if not is_cuda and test_info.attention_ex == ref_info.attention_ex:
        print(f"    OK: under-test is the reference{note or ' (no oracle at this size)'}")
        return True

    ret, outs, kvf = run_attn(test_info, "ex", is_cuda, cuda, c, "exact")
    if ret == -1:
        print("    SKIP: test attention_ex returned -1 (geometry, weight type or KV tier unsupported)")
        return True
    if ret != 0:
        print(f"    FAIL: test attention_ex returned {ret}")
        return False
    worst, msg = _compare(names, ref, attn_snaps(c, outs, kvf), tol)
    if msg:
        print(f"    FAIL: {msg}")
        return False
    print(f"    OK: max_abs_diff={worst:.2e} (threshold {tol:.0e}){note}")
    return True


# ── Reference-kernel contract: bad input is an error code, never a crash or a write ──

def verify_reference_contract(ref_info, cuda) -> bool:
    """The generic reference must validate every tensor and geometry: -1 for an unserviceable
    one, -2 for a too-small scratch, a provided scratch must give the same bits as malloc,
    and in no error case may out / conv_hist / state / the KV cache change."""
    print("\n  [reference contract] linear_attn / attention_ex validation and scratch handling")
    if not (ref_info.linear_attn and ref_info.attention_ex) or looks_cuda(ref_info):
        print("    SKIP: reference has no host linear_attn / attention_ex")
        return True
    bad = []

    def expect(label, got, want):
        if got != want:
            bad.append(f"{label}: returned {got}, expected {want}")

    g = GdnCase(ref_info, cuda, 2, 4, 16, 3, 32, GDN_WEIGHT_SETS["Q8_0"], 1, seed=3)
    r0, s0 = run_gdn(ref_info, False, cuda, g, "none")
    r1, s1 = run_gdn(ref_info, False, cuda, g, "exact")
    expect("gdn scratch=none", r0, 0)
    expect("gdn scratch=exact", r1, 0)
    if s0 != s1:
        bad.append("gdn: result with a provided scratchpad differs from the malloc path")
    untouched = (unf32(f32b(g.out_init[0])), [0.0] * (g.n_vh * g.S * g.S), [0.0] * ((g.K - 1) * g.C))
    gdn_bad_calls = (
        ("gdn scratch one float short", "short", None, -2),
        ("gdn n_vh % n_kh != 0", "none", lambda b, p, x: setattr(p, "n_vh", 3), -1),
        ("gdn d_conv = 0", "none", lambda b, p, x: setattr(p, "d_conv", 0), -1),
        ("gdn head_dim huge", "none", lambda b, p, x: setattr(p, "head_dim", 1 << 30), -1),
        ("gdn w_qkv rows off by one", "none", lambda b, p, x: b["w_qkv"].t.__setattr__("ne", (g.d_model, g.C - 1, 1, 1)), -1),
        ("gdn x is F16", "none", lambda b, p, x: setattr(b["x"].t, "kind", SPITE_TYPE_F16), -1),
        ("gdn w_gate type id unknown", "none", lambda b, p, x: setattr(b["w_gate"].t, "kind", 4), -1),
        ("gdn w_out not contiguous", "none", lambda b, p, x: b["w_out"].t.nb.__setitem__(1, b["w_out"].t.nb[1] + 2), -1),
        ("gdn ssm_a NULL", "none", lambda b, p, x: b.__setitem__("ssm_a", None), -1),
        ("gdn conv_w too small", "none", lambda b, p, x: b["conv_w"].t.__setattr__("ne", (g.K, g.C - 1, 1, 1)), -1),
        ("gdn state is F16", "none", lambda b, p, x: setattr(b["state"].t, "kind", SPITE_TYPE_F16), -1),
    )
    for label, mode, tamper, want in gdn_bad_calls:
        ret, snaps = run_gdn(ref_info, False, cuda, g, mode, tamper)
        expect(label, ret, want)
        if ret == want and snaps and snaps[0] != untouched:
            bad.append(f"{label}: an error return modified out / state / conv_hist")

    a = AttnCase(ref_info, cuda, 64, 4, 2, 16, 8, 1, 12, 5, 1, ATTN_WEIGHT_SETS["Q8_0"], seed=5)
    r0, o0, k0 = run_attn(ref_info, "ex", False, cuda, a, "none")
    r1, o1, k1 = run_attn(ref_info, "ex", False, cuda, a, "exact")
    expect("attn scratch=none", r0, 0)
    expect("attn scratch=exact", r1, 0)
    if (o0, k0) != (o1, k1):
        bad.append("attn: result with a provided scratchpad differs from the malloc path")
    # the generic op's own work area: qfull + k + att + scores (it needs less than the ABI's formula)
    need = a.nh * a.q_stride + a.kv_stride + a.nh * a.hd + a.nh * (a.pos0 + 1)
    r2, o2, k2 = run_attn(ref_info, "ex", False, cuda, a, "exact", scratch_floats=need)
    expect("attn scratch exactly its own need", r2, 0)
    if (o0, k0) != (o2, k2):
        bad.append("attn: result with a minimal scratchpad differs from the malloc path")
    attn_bad_calls = (
        ("attn scratch one float short", "short", None, -2),
        ("attn pos == n_ctx", "none", lambda b, p, x: setattr(x, "pos", a.n_ctx), -2),
        ("attn pos < 0", "none", lambda b, p, x: setattr(x, "pos", -1), -2),
        ("attn rope_dim odd", "none", lambda b, p, x: setattr(p, "rope_dim", 7), -1),
        ("attn rope_dim > head_dim", "none", lambda b, p, x: setattr(p, "rope_dim", 18), -1),
        ("attn rope_dim = 0", "none", lambda b, p, x: setattr(p, "rope_dim", 0), -1),
        ("attn head_dim does not match wq", "none", lambda b, p, x: setattr(p, "head_dim", 8), -1),
        ("attn gated_q mismatches wq rows", "none", lambda b, p, x: setattr(p, "gated_q", 0), -1),
        ("attn gated_q = 2", "none", lambda b, p, x: setattr(p, "gated_q", 2), -1),
        ("attn n_heads % n_kv_heads != 0", "none", lambda b, p, x: setattr(x, "n_kv_heads", 3), -1),
        ("attn n_heads = 0", "none", lambda b, p, x: setattr(x, "n_heads", 0), -1),
        ("attn wo rows != out", "none", lambda b, p, x: b["wo"].t.__setattr__("ne", (a.nhd, a.d_model - 1, 1, 1)), -1),
        ("attn wv type id unknown", "none", lambda b, p, x: setattr(b["wv"].t, "kind", 4), -1),
        ("attn q_norm too short", "none", lambda b, p, x: b["q_norm"].t.__setattr__("ne", (a.hd - 2, 1, 1, 1)), -1),
        ("attn KV cache F16", "none", lambda b, p, x: (setattr(b["k"].t, "kind", SPITE_TYPE_F16), setattr(b["v"].t, "kind", SPITE_TYPE_F16)), -1),
        ("attn KV row width wrong", "none", lambda b, p, x: b["k"].t.__setattr__("ne", (a.kv_stride - 2, a.n_ctx, 1, 1)), -1),
        ("attn wk NULL", "none", lambda b, p, x: b.__setitem__("wk", None), -1),
    )
    for label, mode, tamper, want in attn_bad_calls:
        ret, outs, kvf = run_attn(ref_info, "ex", False, cuda, a, mode, tamper, scratch_floats=need)
        expect(label, ret, want)
        if ret == want and outs and outs[0] != unf32(f32b(a.out_init[0])):
            bad.append(f"{label}: an error return modified out")

    if bad:
        for m in bad:
            print(f"    FAIL: {m}")
        return False
    print(f"    OK: {len(gdn_bad_calls)} GDN + {len(attn_bad_calls)} attention_ex malformed calls "
          f"rejected cleanly, provided-scratch path bit-identical to malloc")
    return True


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify a spite kernel .so")
    parser.add_argument("kernel", help="Path to the kernel .so/.dylib to verify")
    parser.add_argument("--model", help="Optional: path to a .gguf model file")
    parser.add_argument("--ref", default=None,
                        help="Path to reference kernel .so (default: build generic)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.abspath(os.path.join(script_dir, "../.."))

    ref_path = args.ref
    if not ref_path:
        # Check if already built in build tree
        cmake_ref = os.path.join(repo_root, "build/kernels/generic/generic/libkernel_generic.so")
        if os.path.exists(cmake_ref):
            ref_path = cmake_ref
        else:
            ref_path = "/tmp/libkernel_generic_ref.so"
            print("Building generic reference kernel...")
            ret = os.system(
                f"cc -std=c11 -O2 -fPIC -shared -I'{repo_root}' "
                f"'{repo_root}/core/quant.c' "
                f"'{repo_root}/kernels/generic/generic/ops.c' "
                f"'{repo_root}/kernels/generic/generic/linear_attn.c' "
                f"'{repo_root}/kernels/generic/generic/kernel.c' "
                f"-lm -o '{ref_path}'"
            )
            if ret != 0:
                print("ERROR: failed to build reference kernel")
                sys.exit(1)

    print(f"\nReference : {ref_path}")
    print(f"Under test: {args.kernel}")

    ref_lib  = load_kernel(ref_path)
    test_lib = load_kernel(args.kernel)

    print("\n── ABI check ──────────────────────────────────────────────────────")
    ref_info = check_abi(ref_lib,  "reference")
    ref_info._lib = ref_lib          # lets the v7 layer cases decode packed weights for their oracle
    test_info = check_abi(test_lib, "under-test")

    cuda = CudaHelper()
    gpu_arch = (test_info.gpu_arch or b"").lower()
    is_cuda = b"sm_" in gpu_arch or b"cuda" in gpu_arch or b"nvidia" in gpu_arch

    print("\n── Op correctness ─────────────────────────────────────────────────")
    passed = True
    passed &= verify_rms_norm(ref_info, test_info, cuda, is_cuda, dim=64)
    passed &= verify_rms_norm(ref_info, test_info, cuda, is_cuda, dim=4096)

    passed &= verify_matmul(ref_info, test_info, cuda, is_cuda, cols=64, rows=128)
    passed &= verify_matmul(ref_info, test_info, cuda, is_cuda, cols=4096, rows=4096)

    # Q8_0 is the format the shipped model actually uses for every projection,
    # so the F32 cases above never touch the decode matvec.  These column counts
    # step through a K-split kernel's instantiations (512 -> 2 warps per row,
    # 1024 -> 4, 2048/4096 -> 8) and 64 covers the staged one-warp-per-row path.
    if SPITE_TYPE_Q8_0 in tuple(test_info.supported_quants):
        for cols_q8 in (64, 512, 1024, 2048, 4096):
            passed &= verify_matmul(ref_info, test_info, cuda, is_cuda,
                                    cols=cols_q8, rows=256, q8=True)
        # Rows above the row-axis crossover take the staged one-warp-per-row
        # kernel instead of a K-split, so pin that shape too (the column count
        # is deliberately small: it is the row count that selects the path).
        passed &= verify_matmul(ref_info, test_info, cuda, is_cuda,
                                cols=64, rows=16384, q8=True)
    else:
        print("\n  [matmul] q8_0 shapes: SKIP - under-test kernel does not declare Q8_0")

    # Every SpiteType, whatever the kernel declares: supported_quants has only 8
    # slots (0-terminated, F32 == 0 unlistable) so it cannot enumerate a library
    # that decodes all 28.  Just try the op: -1 for an undeclared type is SKIP,
    # a mismatch is FAIL.  Two shapes: many rows, and an odd block count.
    tally = {"ok": 0, "skip": 0, "fail": 0}
    for kind, _name, _bb, _be in SPITE_TYPES:
        for rows_q, cols_q in ((256, 1024), (37, 768)):
            tally[verify_matmul_quant(ref_info, test_info, cuda, is_cuda, kind, rows_q, cols_q)] += 1
    print(f"\n  [matmul] per-type summary over {len(SPITE_TYPES)} types x 2 shapes: "
          f"{tally['ok']} OK, {tally['skip']} SKIP, {tally['fail']} FAIL")
    passed &= tally["fail"] == 0

    passed &= verify_ffn(ref_info, test_info, cuda, is_cuda, hidden=64, ffn_dim=128)
    passed &= verify_ffn(ref_info, test_info, cuda, is_cuda, hidden=2048, ffn_dim=4096)

    # head_dim 16 and 80 are outside every GPU kernel's flash dispatch, so these
    # pin the portable VBR back end at both ends of its range; the rest pin the
    # flash back end: one tile plus a tail, a non-power-of-two head_dim whose
    # lanes own three dims each, and a multi-tile history that splits across
    # chunks.
    passed &= verify_attention(ref_info, test_info, cuda, is_cuda, use_qk_norm=True)
    passed &= verify_attention(ref_info, test_info, cuda, is_cuda, use_qk_norm=False)
    passed &= verify_attention(ref_info, test_info, cuda, is_cuda, head_dim=80, n_ctx=160, pos=20,
                               use_qk_norm=True)
    passed &= verify_attention(ref_info, test_info, cuda, is_cuda, head_dim=64, n_ctx=96, pos=7,
                               use_qk_norm=True)
    passed &= verify_attention(ref_info, test_info, cuda, is_cuda, head_dim=96, n_ctx=160, pos=20,
                               use_qk_norm=True)
    passed &= verify_attention(ref_info, test_info, cuda, is_cuda, head_dim=64, n_ctx=320, pos=140,
                               use_qk_norm=True)

    # VBR cache tiers.  The generic C reference declares F32-only KV, so with the
    # default reference these report SKIP; pass --ref <tier-capable kernel.so>
    # (e.g. a GPU .so from before the change) and they become a differential
    # check that both kernels decode identical cache bytes identically — the
    # shipped reference cannot check the block codecs by construction.
    if declared_kv_tiers(ref_info) & ((1 << SPITE_TYPE_F16) | (1 << SPITE_TYPE_Q8_0) |
                                      (1 << SPITE_TYPE_Q5_1) | (1 << SPITE_TYPE_Q4_0)):
        for tier in (SPITE_TYPE_F16, SPITE_TYPE_Q8_0, SPITE_TYPE_Q5_1, SPITE_TYPE_Q4_0):
            passed &= verify_attention(ref_info, test_info, cuda, is_cuda, head_dim=64, n_ctx=160,
                                       pos=140, use_qk_norm=True, kv_kind=tier)
    else:
        print("\n  [attention] VBR tiers f16/q8_0/q5_1/q4_0 (block decoders): SKIP — the "
              "reference\n              kernel declares F32-only KV; pass a tier-capable .so "
              "as --ref to check them")

    # ── ABI v7 layer ops (hybrid Qwen3.5-style models) ──
    # The reference first has to hold up against bad input; then each case checks the
    # reference against an independent float64 model (smallest geometries) and the kernel
    # under test against the reference.  Ops a kernel leaves NULL report SKIP.
    passed &= verify_reference_contract(ref_info, cuda)

    # Gated Delta Net layer, three tokens each (state / conv history carry between calls):
    # F32 weights, Q8_0, and Q4_K/Q5_K/Q6_K.  The last geometry is the 27B shape; its
    # projections are 30-55 MB each, so it runs Q8_0 only (and has no python oracle).
    for geom in ((2, 4, 64, 4, 256), (4, 8, 128, 4, 512)):
        for wset in GDN_WEIGHT_SETS:
            passed &= verify_linear_attn(ref_info, test_info, cuda, is_cuda, *geom, wset=wset)
    passed &= verify_linear_attn(ref_info, test_info, cuda, is_cuda, 16, 48, 128, 4, 5120,
                                 wset="Q8_0", oracle=False)
    # small corners: no history (K=1), K=2, n_vh == n_kh, 3 value heads per key head
    for geom in ((1, 2, 16, 1, 32), (2, 2, 16, 2, 64), (2, 6, 16, 3, 64)):
        passed &= verify_linear_attn(ref_info, test_info, cuda, is_cuda, *geom)

    # Extended attention, three sequential tokens against a pre-filled KV cache.
    # (heads, kv_heads, head_dim, rope_dim, gated, ctx, pos0, d_model)
    for wset in ATTN_WEIGHT_SETS:
        passed &= verify_attention_ex(ref_info, test_info, cuda, is_cuda, d_model=256, n_heads=4,
                                      n_kv_heads=2, head_dim=64, rope_dim=32, gated=1,
                                      n_ctx=48, pos=17, wset=wset)
        passed &= verify_attention_ex(ref_info, test_info, cuda, is_cuda, d_model=512, n_heads=8,
                                      n_kv_heads=2, head_dim=256, rope_dim=64, gated=1,
                                      n_ctx=96, pos=40, wset=wset)
    # the other gated / partial-RoPE combinations, and the plain case that must equal `attention`
    passed &= verify_attention_ex(ref_info, test_info, cuda, is_cuda, head_dim=64, rope_dim=64,
                                  gated=1)                                   # gate, full RoPE
    passed &= verify_attention_ex(ref_info, test_info, cuda, is_cuda, head_dim=64, rope_dim=16,
                                  gated=0)                                   # partial RoPE, no gate
    for wset in ATTN_WEIGHT_SETS:
        passed &= verify_attention_ex(ref_info, test_info, cuda, is_cuda, head_dim=64, rope_dim=64,
                                      gated=0, wset=wset)                    # == attention
    passed &= verify_attention_ex(ref_info, test_info, cuda, is_cuda, head_dim=64, rope_dim=64,
                                  gated=0, use_qk_norm=False)
    passed &= verify_attention_ex(ref_info, test_info, cuda, is_cuda, d_model=256, n_heads=8,
                                  n_kv_heads=2, head_dim=128, rope_dim=128, gated=0, n_ctx=160,
                                  pos=140)
    # VBR cache tiers need a tier-capable --ref (the generic reference is F32-only), as above.
    if declared_kv_tiers(ref_info) & ((1 << SPITE_TYPE_F16) | (1 << SPITE_TYPE_Q8_0) |
                                      (1 << SPITE_TYPE_Q5_1) | (1 << SPITE_TYPE_Q4_0)):
        for tier in (SPITE_TYPE_F16, SPITE_TYPE_Q8_0, SPITE_TYPE_Q5_1, SPITE_TYPE_Q4_0):
            passed &= verify_attention_ex(ref_info, test_info, cuda, is_cuda, head_dim=64,
                                          rope_dim=32, gated=1, n_ctx=160, pos=140,
                                          kv_kind=tier)
    else:
        print("\n  [attention_ex] VBR tiers f16/q8_0/q5_1/q4_0: SKIP - the reference kernel "
              "declares F32-only KV; pass a tier-capable .so as --ref to check them")

    print("\n── Result ─────────────────────────────────────────────────────────")
    if passed:
        print("PASSED — all checks OK")
        sys.exit(0)
    else:
        print("FAILED — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()

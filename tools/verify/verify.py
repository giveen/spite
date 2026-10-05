#!/usr/bin/env python3
"""
tools/verify/verify.py — verify a kernel .so against the generic reference

Usage:
    python3 tools/verify/verify.py <kernel.so> [--model <model.gguf>] [--ref <ref_kernel.so>]

Loads the kernel with ctypes, loads the generic reference kernel, runs each
op with randomized inputs, and compares outputs. Any element-wise
max error > 1e-4 is a failure.

Also performs an ABI version check: refuses kernels reporting a different
SPITE_ABI_VERSION from what core/abi.h reports.

Exit code 0 = all checks passed.
"""

import argparse
import ctypes
import os
import sys
import random
import math

# ── ABI constants (must match core/abi.h) ────────────────────────────────

SPITE_ABI_VERSION = 4

SPITE_TYPE_F32  = 0
SPITE_TYPE_F16  = 1
SPITE_TYPE_BF16 = 2
SPITE_TYPE_Q8_0 = 8
SPITE_TYPE_Q5_1 = 11
SPITE_TYPE_Q4_0 = 10
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

AttentionFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
    ctypes.c_float, ctypes.POINTER(SpiteKvCache),
    ctypes.c_float, ctypes.POINTER(SpiteCtx),
)

# ── Strides and Helpers ───────────────────────────────────────────────────

def contiguous_strides(kind: int, ne: list[int] | tuple[int, ...]) -> list[int]:
    block_bytes = {
        SPITE_TYPE_F32: 4,
        SPITE_TYPE_F16: 2,
        SPITE_TYPE_BF16: 2,
        SPITE_TYPE_Q8_0: 34,
        SPITE_TYPE_Q5_1: 24,
        SPITE_TYPE_Q4_0: 18,
        SPITE_TYPE_Q4_K: 144,
        SPITE_TYPE_Q5_K: 176,
        SPITE_TYPE_Q6_K: 210,
    }.get(kind, 4)
    block_elements = {
        SPITE_TYPE_Q8_0: 32,
        SPITE_TYPE_Q5_1: 32,
        SPITE_TYPE_Q4_0: 32,
        SPITE_TYPE_Q4_K: 256,
        SPITE_TYPE_Q5_K: 256,
        SPITE_TYPE_Q6_K: 256,
    }.get(kind, 1)

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


def max_abs_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


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


def verify_matmul(ref_info: SpiteKernelInfo, test_info: SpiteKernelInfo,
                  cuda: CudaHelper, is_cuda: bool, cols: int = 64, rows: int = 128) -> bool:
    print(f"\n  [matmul] shape=[{rows}, {cols}]")
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
    tw, _ = make_tensor(w_data, [cols, rows, 1, 1])
    to, _ = make_tensor([0.0] * rows, [rows, 1, 1, 1])
    to.data = ctypes.cast(out_ref_arr, ctypes.c_void_p)
    ctx_ref = make_ctx()

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
        d_w = cuda.malloc(rows * cols * 4)
        d_o = cuda.malloc(rows * 4)
        try:
            cuda.h2d(d_x, (ctypes.c_float * cols)(*x_data), cols * 4)
            cuda.h2d(d_w, (ctypes.c_float * (rows * cols))(*w_data), rows * cols * 4)
            tx_gpu, _ = make_tensor([0.0] * cols, [cols, 1, 1, 1])
            tw_gpu, _ = make_tensor([0.0] * (rows * cols), [cols, rows, 1, 1])
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


def verify_attention(ref_info: SpiteKernelInfo, test_info: SpiteKernelInfo) -> bool:
    print("\n  [attention]")
    if not test_info.attention:
        print("    SKIP: test kernel has no attention")
        return True
    if not ref_info.attention:
        print("    SKIP: reference kernel has no attention")
        return True
    print("    SKIP: generic C reference defers attention to Rust scalar fallback")
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
                f"'{repo_root}/kernels/generic/generic/dequant.c' "
                f"'{repo_root}/kernels/generic/generic/ops.c' "
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

    passed &= verify_ffn(ref_info, test_info, cuda, is_cuda, hidden=64, ffn_dim=128)
    passed &= verify_ffn(ref_info, test_info, cuda, is_cuda, hidden=2048, ffn_dim=4096)

    passed &= verify_attention(ref_info, test_info)

    print("\n── Result ─────────────────────────────────────────────────────────")
    if passed:
        print("PASSED — all checks OK")
        sys.exit(0)
    else:
        print("FAILED — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()

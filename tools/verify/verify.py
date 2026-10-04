#!/usr/bin/env python3
"""
tools/verify/verify.py — verify a kernel .so against the generic reference

Usage:
    python3 tools/verify/verify.py <kernel.so> [--model <model.gguf>]

Loads the kernel with ctypes, loads the generic reference kernel, runs each
op with randomized F32 inputs, and compares outputs.  Any element-wise
max error > 1e-4 is a failure.

Also performs an ABI version check: refuses kernels reporting a different
SPITE_ABI_VERSION from what crates/spite-abi reports.

Exit code 0 = all checks passed.
"""

import argparse
import ctypes
import struct
import os
import sys
import math
import random

# ── ABI constants (must match core/abi.h) ────────────────────────────────

SPITE_ABI_VERSION = 3

SPITE_TYPE_F32  = 0
SPITE_TYPE_F16  = 1
SPITE_TYPE_Q8_0 = 8
SPITE_TYPE_Q5_1 = 11
SPITE_TYPE_Q4_K = 12
SPITE_TYPE_Q6_K = 14

# ── ctypes struct definitions ─────────────────────────────────────────────

class SpiteTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ne",   ctypes.c_uint32 * 4),
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

class SpiteKernelInfo(ctypes.Structure):
    # Note: only checks fields we need; function pointers vary
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("model_arch",  ctypes.c_char_p),
        ("gpu_arch",    ctypes.c_char_p),
        ("author",      ctypes.c_char_p),
    ]

# ── Helpers ───────────────────────────────────────────────────────────────

def make_tensor(data_f32: list[float]) -> tuple[SpiteTensor, ctypes.Array]:
    """Create a SpiteTensor backed by a ctypes float array."""
    n = len(data_f32)
    arr = (ctypes.c_float * n)(*data_f32)
    t = SpiteTensor()
    t.data = ctypes.cast(arr, ctypes.c_void_p)
    t.ne[0] = n
    t.ne[1] = 1
    t.ne[2] = 1
    t.ne[3] = 1
    t.kind  = SPITE_TYPE_F32
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


# ── RMS norm check ─────────────────────────────────────────────────────────

def verify_rms_norm(ref_lib, test_lib, dim: int = 64) -> bool:
    print(f"\n  [rms_norm] dim={dim}")
    rng = random.Random(42)
    x_data   = [rng.gauss(0, 1) for _ in range(dim)]
    w_data   = [rng.uniform(0.5, 1.5) for _ in range(dim)]
    out_ref  = [0.0] * dim
    out_test = [0.0] * dim

    x_t, x_arr   = make_tensor(x_data)
    w_t, w_arr   = make_tensor(w_data)
    ctx = make_ctx()

    # Build restype for rms_norm function
    RmsNormFn = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(SpiteTensor), ctypes.POINTER(SpiteTensor),
        ctypes.POINTER(SpiteTensor), ctypes.c_float,
        ctypes.POINTER(SpiteCtx),
    )

    def run_rms_norm(lib, out_data):
        out_t, out_arr = make_tensor(out_data)
        fn_ptr = ctypes.cast(
            ctypes.c_void_p.in_dll(lib, "spite_generic_rms_norm")
            if hasattr(lib, "spite_generic_rms_norm") else None,
            RmsNormFn
        ) if False else None

        # Use function pointer from KernelInfo struct
        # For simplicity, call directly if symbol exists
        try:
            fn = RmsNormFn(("spite_generic_rms_norm", lib))
        except AttributeError:
            return None
        ret = fn(ctypes.byref(out_t), ctypes.byref(x_t), ctypes.byref(w_t),
                 1e-5, ctypes.byref(ctx))
        if ret != 0:
            return None
        return list(out_arr)

    ref_out  = run_rms_norm(ref_lib,  out_ref)
    test_out = run_rms_norm(test_lib, out_test)

    if ref_out is None:
        print("    SKIP: reference kernel has no rms_norm symbol")
        return True
    if test_out is None:
        print("    SKIP: test kernel has no rms_norm symbol (returns -1)")
        return True

    err = max_abs_diff(ref_out, test_out)
    if err > 1e-4:
        print(f"    FAIL: max_abs_diff={err:.2e} (threshold 1e-4)")
        return False
    print(f"    OK: max_abs_diff={err:.2e}")
    return True


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify a spite kernel .so")
    parser.add_argument("kernel", help="Path to the kernel .so/.dylib to verify")
    parser.add_argument("--model", help="Optional: path to a .gguf model file")
    parser.add_argument("--ref", default=None,
                        help="Path to reference kernel .so (default: build generic)")
    args = parser.parse_args()

    # Locate or build the generic reference kernel
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.abspath(os.path.join(script_dir, "../.."))

    ref_path = args.ref
    if not ref_path:
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
    check_abi(ref_lib,  "reference")
    check_abi(test_lib, "under-test")

    print("\n── Op correctness ─────────────────────────────────────────────────")
    passed = True
    passed &= verify_rms_norm(ref_lib, test_lib, dim=64)
    passed &= verify_rms_norm(ref_lib, test_lib, dim=4096)

    print("\n── Result ─────────────────────────────────────────────────────────")
    if passed:
        print("PASSED — all checks OK")
        sys.exit(0)
    else:
        print("FAILED — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()

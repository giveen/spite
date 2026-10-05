#!/usr/bin/env python3
"""
tools/benchmark/bench.py — wrapper around the `spite-bench` binary

Usage:
    python3 tools/benchmark/bench.py --kernel <kernel.so> --model <model.gguf>
                                      [--n-runs 10] [--n-tokens 512]
                                      [--device auto|cuda|cpu] [--card <card>]
                                      [--kernels-dir <dir>] [--json out.json]

Invokes `spite-bench` and pretty-prints the results.
Exit code 0 = benchmark completed successfully.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time


def inspect_kernel_arch(kernel_path: str) -> str | None:
    """Attempt to read gpu_arch from spite_kernel_info in the kernel library."""
    if not kernel_path or not os.path.exists(kernel_path):
        return None
    try:
        class SpiteKernelInfo(ctypes.Structure):
            _fields_ = [
                ("abi_version", ctypes.c_uint32),
                ("model_arch",  ctypes.c_char_p),
                ("gpu_arch",    ctypes.c_char_p),
                ("author",      ctypes.c_char_p),
            ]
        lib = ctypes.CDLL(kernel_path)
        lib.spite_kernel_info.restype = ctypes.POINTER(SpiteKernelInfo)
        lib.spite_kernel_info.argtypes = []
        info = lib.spite_kernel_info()
        if info and info.contents.gpu_arch:
            return info.contents.gpu_arch.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return None


def run_bench(model: str, n_runs: int, n_tokens: int = 512,
              device: str = "auto", card: str | None = None,
              gpu_arch: str | None = None, kernels_dir: str | None = None) -> dict:
    """Invoke `spite-bench` for the given model and return parsed results."""
    cmd = [
        "cargo", "run", "--release", "-p", "spite-bench", "--",
        "--model", model,
        "--n-runs", str(n_runs),
        "--n-tokens", str(n_tokens),
        "--device", device,
    ]
    if card:
        cmd.extend(["--card", card])
    if gpu_arch:
        cmd.extend(["--gpu-arch", gpu_arch])
    if kernels_dir:
        cmd.extend(["--kernels-dir", kernels_dir])

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        print("ERROR running spite-bench:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    return {
        "model": model,
        "n_runs": n_runs,
        "n_tokens": n_tokens,
        "elapsed_s": round(elapsed, 2),
        "stdout": result.stdout,
    }


def main():
    parser = argparse.ArgumentParser(description="spite kernel benchmark wrapper")
    parser.add_argument("--kernel",      default=None, help="Path to kernel .so (optional)")
    parser.add_argument("--model",       required=True, help="Path to .gguf model")
    parser.add_argument("--n-runs",      type=int, default=5)
    parser.add_argument("--n-tokens",    type=int, default=512)
    parser.add_argument("--device",      choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--card",        default=None, help="GPU card name (e.g. RTX_5090)")
    parser.add_argument("--gpu-arch",    default=None, help="GPU arch override (e.g. sm_120)")
    parser.add_argument("--kernels-dir", default=None, help="Path to kernels directory")
    parser.add_argument("--json",        default=None, help="Write JSON output here")
    args = parser.parse_args()

    device = args.device
    gpu_arch = args.gpu_arch
    kernels_dir = args.kernels_dir

    if args.kernel:
        k_arch = inspect_kernel_arch(args.kernel)
        if k_arch:
            if not gpu_arch and k_arch != "generic":
                gpu_arch = k_arch
            if device == "auto" and "sm_" in k_arch:
                device = "cuda"

    print(f"Model:        {args.model}")
    print(f"Device:       {device}")
    if gpu_arch:
        print(f"GPU arch:     {gpu_arch}")
    if args.card:
        print(f"Card:         {args.card}")
    if args.kernel:
        print(f"Kernel:       {args.kernel}")
    print(f"Runs:         {args.n_runs} (tokens: {args.n_tokens})")
    print()

    result = run_bench(
        args.model,
        args.n_runs,
        n_tokens=args.n_tokens,
        device=device,
        card=args.card,
        gpu_arch=gpu_arch,
        kernels_dir=kernels_dir,
    )
    print(result["stdout"])
    print(f"Elapsed: {result['elapsed_s']}s")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()


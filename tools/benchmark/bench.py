#!/usr/bin/env python3
"""
tools/benchmark/bench.py — wrapper around the `spite-bench` binary

Usage:
    python3 tools/benchmark/bench.py --kernel <kernel.so> --model <model.gguf>
                                      [--n-runs 10] [--json out.json]

Invokes `spite-bench` and pretty-prints the results.
The --kernel argument is accepted for interface compatibility but is not
yet forwarded (the harness currently benchmarks whole-model throughput).
Exit code 0 = benchmark completed successfully.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def run_bench(model: str, n_runs: int) -> dict:
    """Invoke `spite-bench` for the given model and return parsed results."""
    cmd = ["cargo", "run", "--release", "-p", "spite-bench", "--",
           "--model", model, "--n-runs", str(n_runs)]
    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        print("ERROR running spite-bench:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    return {
        "model":  model,
        "n_runs": n_runs,
        "elapsed_s": round(elapsed, 2),
        "stdout": result.stdout,
    }


def main():
    parser = argparse.ArgumentParser(description="spite kernel benchmark wrapper")
    parser.add_argument("--kernel",  required=True, help="Path to kernel .so")
    parser.add_argument("--model",   required=True, help="Path to .gguf model")
    parser.add_argument("--n-runs",  type=int, default=10)
    parser.add_argument("--json",    default=None, help="Write JSON output here")
    args = parser.parse_args()

    print(f"Benchmarking: {args.kernel}")
    print(f"Model:        {args.model}")
    print(f"Runs:         {args.n_runs}")
    print()

    result = run_bench(args.model, args.n_runs)
    print(result["stdout"])
    print(f"Elapsed: {result['elapsed_s']}s")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()

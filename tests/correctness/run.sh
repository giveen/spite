#!/usr/bin/env bash
# tests/correctness/run.sh — run the correctness test suite
#
# Usage:
#   ./tests/correctness/run.sh [--model <path>] [--kernels <dir>]
#
# Without arguments, runs Rust unit tests only (no model file needed).
# With --model, also runs end-to-end correctness checks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL=""
KERNELS_DIR="${REPO_ROOT}/build/kernels"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";    shift 2 ;;
        --kernels)  KERNELS_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=== spite correctness tests ==="
echo "repo:    ${REPO_ROOT}"
echo "kernels: ${KERNELS_DIR}"
echo ""

# ── 1. Rust unit tests ─────────────────────────────────────────────────────
echo "[1/3] Running Rust unit tests..."
cargo test --workspace --manifest-path "${REPO_ROOT}/Cargo.toml" 2>&1
echo "  ✓ Rust unit tests passed"

# ── 2. Generic C kernel: build and smoke-test ──────────────────────────────
echo ""
echo "[2/3] Building and verifying generic C kernel..."

TMP_SO="$(mktemp /tmp/libkernel_generic_XXXXX.so)"
trap 'rm -f "$TMP_SO"' EXIT

cc -std=c11 -Wall -Wextra -O2 -fPIC -shared \
    -I"${REPO_ROOT}" \
    "${REPO_ROOT}/kernels/generic/generic/dequant.c" \
    "${REPO_ROOT}/kernels/generic/generic/ops.c" \
    "${REPO_ROOT}/kernels/generic/generic/kernel.c" \
    -lm -o "$TMP_SO"

# Verify the ABI entry point is exported
if ! nm -D "$TMP_SO" | grep -q "spite_kernel_info"; then
    echo "ERROR: spite_kernel_info not exported from generic kernel" >&2
    exit 1
fi
echo "  ✓ Generic kernel builds and exports spite_kernel_info"

# ── 3. Model-based tests (optional) ───────────────────────────────────────
if [[ -n "$MODEL" ]]; then
    echo ""
    echo "[3/3] Running model correctness checks against: $MODEL"
    # TODO: implement per-op correctness checks using spite-compute reference
    # Expected checks:
    #   - Q8_0 roundtrip: quantize → dequantize, max error < 0.5
    #   - Q4_K roundtrip: similar
    #   - RMS norm: compare C kernel output vs Rust reference
    #   - Attention: compare GQA outputs at known positions
    echo "  (model tests not yet implemented — contribute them!)"
else
    echo ""
    echo "[3/3] Skipping model tests (no --model provided)"
fi

echo ""
echo "=== All correctness tests passed ==="

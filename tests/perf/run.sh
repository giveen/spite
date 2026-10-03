#!/usr/bin/env bash
# tests/perf/run.sh — run the performance regression test suite
#
# Usage:
#   ./tests/perf/run.sh --model <path> [--baseline <json>] [--out <json>]
#
# Records tokens/sec and TTFT for a set of prompt lengths.
# Fails if any metric regresses more than 5% from --baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL=""
BASELINE=""
OUT="/tmp/spite_perf_$(date +%s).json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";    shift 2 ;;
        --baseline) BASELINE="$2"; shift 2 ;;
        --out)      OUT="$2";      shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 --model <path> [--baseline <json>] [--out <json>]" >&2
    exit 1
fi

echo "=== spite performance tests ==="
echo "model: ${MODEL}"
echo "out:   ${OUT}"

# TODO: implement using spite-bench once the inference loop is wired up
# Expected benchmark matrix:
#   prompt_len  × n_tokens  → tps, ttft_ms
#   8           × 128
#   128         × 128
#   1024        × 256
#   4096        × 128

echo ""
echo "(performance tests not yet implemented — contribute them!)"
echo "Planned entry point: cargo run -p spite-bench -- --model $MODEL --json"
echo ""
echo "See crates/spite-bench/src/main.rs for the harness structure."

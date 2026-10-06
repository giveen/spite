# Multi-GPU

spite can spread one model over several CUDA GPUs in the same machine by
**pipeline (layer) splitting**: each GPU holds a contiguous block of layers
and runs it, then hands the hidden state to the next GPU. Use it when a model
does not fit in one card's VRAM, e.g. a 27B Qwen3.8 Q5_K_S (~17.4 GiB) on
16 GB Tesla P100s.

---

## What is supported

| | Status |
|---|---|
| Hybrid models (Qwen3.5 / Qwen3.8 family, GGUF arch `qwen35`) | Pipeline split ✅ |
| NextN / MTP draft head | Runs on the last GPU ✅ |
| Dense models (`gpu_dense` path: Llama, Qwen3, …) | Single GPU only |
| Tensor parallelism (`spite-parallel::p100_multi`) | Policy only, not wired |
| Mixed vendors (CUDA + HIP) | Not supported |

The split needs **no peer-to-peer access or NVLink**. Only the hidden state
(`d_model` floats, 20 KB for a 27B model) crosses a stage boundary per token,
through host memory, so plain PCIe boxes work. Decode is still sequential:
splitting adds capacity, not speed. The hand-off itself costs nothing
measurable: a 3-stage split on one RTX 5090 ran at 42.6 tok/s vs 42.2 unsplit.

---

## How layers are placed

With no flags, spite considers every visible GPU:

1. If the whole model (weights + KV + recurrent state + scratch + 512 MiB
   headroom) fits on the first GPU, it stays there. Single-GPU behavior is
   unchanged.
2. Otherwise it spreads the layers over **all** visible GPUs in proportion to
   each one's free VRAM, balancing **bytes** rather than layer counts (only
   every 4th Qwen3.5 layer has a KV cache, so counts would be uneven).
3. The last GPU also holds the output norm, LM head (`output.weight`) and the
   NextN/MTP block, so it gets fewer trunk layers.
4. Each GPU's total is checked against its free memory before anything is
   uploaded; a GPU that is too small fails with a message naming it.

The token embedding is looked up on the host, so `token_embd.weight` is not
uploaded at all unless it doubles as the LM head (tied embeddings).

---

## Building for older GPUs (Pascal / Tesla P100)

CUDA 13 dropped `sm_60`. Build P100 kernels with a **CUDA 12.x** toolkit and a
driver branch that still supports Pascal:

```bash
cmake -B build \
  -DSPITE_MODELS="qwen/qwen3_5" \
  -DSPITE_GPU_ARCHS="TESLA_P100" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build --prefix .

# Confirm the kernel really contains sm_60 code:
cuobjdump --list-elf kernels/qwen/qwen3_5/nvidia/libkernel_qwen_qwen3_5_nvidia.so
#   ELF file 1: ...sm_60.cubin
```

If libcudart 13 is first on the library path, point spite at the 12.x
runtime with `SPITE_CUDART=/usr/local/cuda-12.6/lib64/libcudart.so`.

Newer cards build the same way with their own card name
(`-DSPITE_GPU_ARCHS="RTX_4090"`, etc.).

---

## Running

```bash
spite run -m /models/Altworld_Hemmingway-1-Q5_K_S.gguf \
  --card TESLA_P100 --device cuda --ctx 8192 \
  -p "The old man walked to the harbor and"
```

Always pass **`--device cuda`** when testing GPUs. Under the default
`--device auto`, a missing GPU kernel silently falls back to the CPU path and
a 27B model just looks very slow; `--device cuda` turns that into an error.

A split load prints one line per stage (illustrative numbers for 6× P100):

```
device       : Cuda (hybrid) — weights 17.40 GiB + KV 0.54 GiB (F16) + recurrent state 0.15 GiB + ...
  stage      : GPU 0 — layers 0..12 (3.10 GiB)
  stage      : GPU 1 — layers 12..24 (3.08 GiB)
  ...
  stage      : GPU 5 — layers 56..64 (3.21 GiB)
```

No `stage` lines means the model fit on one GPU.

### Choosing GPUs

| Goal | How |
|---|---|
| Use only some GPUs | `CUDA_VISIBLE_DEVICES=0,1,2` or `--gpus 0,1,2` |
| Order the stages | `--gpus 3,2,1,0` (stage order = list order) |
| Force a split even when it fits | `--layer-split 1,1` (one share per GPU) |
| Uneven split | `--layer-split 20,12` → ≈20/32 of the layers on the first GPU |

`--gpus` and `--layer-split` are advanced (hidden from `--help`).
`SPITE_GPUS` is the environment form of `--gpus`. Without `--gpus`,
`--layer-split` with N shares uses the first N visible GPUs. Shares are
relative; every share must be large enough to get at least one layer.

### Longer context

KV memory grows with `--ctx` and lives on the GPUs holding the full-attention
layers. For a 27B Qwen3.8 with F16 KV that is about 68 KB per token across
all GPUs (~0.27 GiB at 4k, ~17 GiB at the 262144 maximum). If a stage
overflows, lower `--ctx`, add GPUs, or rebalance with `--layer-split`.

---

## Benchmarking

`spite-bench` uses the same automatic placement:

```bash
cargo run --release -p spite-bench -- \
  --model /models/Altworld_Hemmingway-1-Q5_K_S.gguf \
  --card TESLA_P100 --device cuda
```

Stage lines go to stderr; the result row is the whole pipeline. Follow the
benchmark gate in `AGENTS.md` when a `.bench` file is part of a kernel PR.

---

## Testing the split on one GPU

Repeating an ordinal puts several stages on one card, which exercises the
hand-off without extra hardware. The output must match the unsplit run:

```bash
spite run -m model.gguf --device cuda --temperature 0 --max-tokens 48 -p "..." > single.txt
spite run -m model.gguf --device cuda --temperature 0 --max-tokens 48 -p "..." \
  --gpus 0,0,0 --layer-split 22,22,21 > split.txt
```

The CPU-backend test `layer_split_matches_single_stage`
(`crates/spite-models/tests/hybrid_mtp.rs`) checks trunk and MTP logits are
bit-identical between a split and an unsplit decoder.

---

## Troubleshooting

| Message | Fix |
|---|---|
| `nvcc fatal: Unsupported gpu architecture 'sm_60'` | Use a CUDA 12.x toolkit (see above) |
| `model needs X GiB VRAM on GPU 0 … only Y GiB free` | Only one GPU visible or chosen: check `CUDA_VISIBLE_DEVICES` / `--gpus` |
| `GPU n needs X GiB for its pipeline stages …` | Lower `--ctx`, add GPUs, or shift layers off GPU n with `--layer-split` |
| `GPU n does not exist (m visible)` | `--gpus` names an ordinal outside `CUDA_VISIBLE_DEVICES` |
| `layer split has N shares for M GPUs` | Give one `--layer-split` share per GPU |
| `--device cuda: no kernel provides every hybrid op …` | Kernels not built/installed for this card; rebuild with the right `SPITE_GPU_ARCHS` |

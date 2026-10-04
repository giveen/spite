<p align="center">
  <img src="screenshots/logo.png" alt="SPITE Logo" width="700" />
</p>

# spite

**One Engine. Your Model. Your Card.**

Most AI inference tools are built around one GPU brand, one cloud, or one company's
model. spite is built around *you*. You pick the model. You pick the hardware. The
engine adapts — not the other way around.

- **One Engine** — a single Rust host that loads any GGUF model and runs it.
  No cloud required, no subscription, no data leaving your machine.
- **Your Model** — any GGUF file works. Download a model once, run it forever.
  Llama 4, DeepSeek-V4, Qwen 3.5, Mistral 4, Gemma 4, GLM-5 — if it's a `.gguf`, spite loads it.
- **Your Card** — kernels are written *for* specific GPUs, not against the lowest
  common denominator. RTX 3060, RX 7800 XT, Intel Arc, Apple M-series. If nobody
  has written a tuned kernel for your card yet, the generic fallback runs. When
  someone does write one — maybe you — every person with that card gets faster.

---

## What it is

spite is a local inference engine. Run large language models on your own hardware,
serve them from your own machine, and connect any OpenAI-compatible app to it.
The models run in private, the GPU work is as fast as the hardware allows, and
anyone can contribute improvements for any card.

**If you just want to run a model:** download a `.gguf` file, point spite at it,
and it runs.

**If you want to serve an API:** `spite-server` speaks the OpenAI protocol.
Your existing tools, UIs, and scripts work without changes.

**If you have a specific GPU and want to make it faster:** contribute a kernel.
You only need to know your GPU — nothing else in the codebase changes.

---

## Modular by design

spite is built on a single rule: **every layer is replaceable without touching any other layer.**

That sounds abstract, so here's what it means in practice:

### Every model is its own module

Kernels are grouped by family and variant: `kernels/llama/llama4/`,
`kernels/deepseek/v4/`, `kernels/qwen/qwen3_5/`, `kernels/mistral/mistral4/`,
`kernels/gemma/gemma4/`. Adding a new model variant means adding a new
`<family>/<model>/` folder. Nothing about the existing models changes. The
dispatcher finds it automatically.

### Every GPU is its own module

`kernels/llama/llama4/sm_89/` is completely separate from `kernels/llama/llama4/rdna3/`.
An RTX 4090 kernel can use FP8 tensor cores. An RX 7900 XTX kernel can exploit
96 MB of Infinity Cache. An Apple M4 kernel can use the Neural Engine. Each gets
what makes it fast, not a watered-down kernel that has to work on everything.

### Every operation is independently tunable

Kernels don't have to implement everything. A kernel that only optimizes
attention leaves FFN and rms_norm to the fallback. You tune the one op that's
your bottleneck. Later, someone else improves FFN. Both improvements stack
automatically — the dispatcher picks the best available kernel for each op
on each GPU.

### Every subsystem is swappable

The sampler, tokenizer, KV cache backend, and offload policy are all
plugin registries. Register a custom sampler for a specific model or task
and the engine uses it. Register a custom KV cache for a memory-constrained
deployment and the scheduler uses it. Nothing needs to be forked.

```rust
let engine = EngineBuilder::new()
    .with_sampler(PluginKey::for_model("llama4"), Box::new(MyGreedySampler))
    .with_cache(PluginKey::default(), Box::new(PagedKvCache::new(vram)))
    .build(ExecutorConfig::default());
```

### Every component is usable standalone

spite is a Rust workspace. You can use just the loader, just the scheduler,
or just the ABI types for kernel development — without pulling in the full
server stack. Build what you need from the pieces that fit.

---

## Quick start

```bash
# Build
cargo build --release

# Run a model
./target/release/spite run \
  --model ~/models/qwen3.5-8b-instruct.Q4_K_M.gguf \
  --prompt "What is the capital of France?"

# Start an API server (OpenAI-compatible)
./target/release/spite-server \
  --model ~/models/qwen3.5-8b-instruct.Q4_K_M.gguf \
  --port 8080
```

Then point any OpenAI-compatible app at `http://localhost:8080`.

---

## Building GPU kernels

The Rust host runs without GPU kernels (using the generic CPU fallback), but
for full speed you'll want to compile the kernels for your GPU.

```bash
# Find your GPU architecture first — `spite dispatch` shows it:
./target/release/spite dispatch -m ~/models/your.gguf --card RTX_4090

# Build kernels for your card
cmake -B build \
  -DSPITE_MODELS="llama/llama4"  \
  -DSPITE_GPU_ARCHS="RTX_4090"   \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build --prefix .
```

The install step copies the compiled kernels into `./kernels/`, which spite
checks automatically. After that, run normally — no extra flags:

```bash
./target/release/spite run \
  --model ~/models/your.gguf \
  --prompt "Hello"
```

**You don't need to build kernels for every model or every GPU.** Build only
the combination you actually use. The fallback covers everything else.

---

## Supported hardware

Every card listed here runs today via the generic fallback. Tuned kernels exist
where the community has contributed them; all other cards fall back to the
generic CPU path automatically — slower, but always correct.

### NVIDIA

| Architecture | GPU arch | Cards |
|---|---|---|
| Ada Lovelace | `sm_89` | RTX 4090, RTX 4080 Super / 4080, RTX 4070 Ti Super / 4070 Ti / 4070 Super / 4070, RTX 4060 Ti / 4060, RTX 4000 / 5000 / 6000 Ada |
| Ampere | `sm_86` | RTX 3090 Ti / 3090 / 3080 Ti / 3080 / 3070 Ti / 3070 / 3060 Ti / 3060, RTX A2000–A6000 |
| Turing | `sm_75` | RTX 2080 Ti / 2080 Super / 2080 / 2070 Super / 2070 / 2060 Super / 2060, GTX 1660 Ti / 1660 Super / 1660 |
| Blackwell | `sm_120` | RTX 5090 / 5080 / 5070 Ti / 5070 / 5060 Ti / 5060 |

### AMD

| Architecture | GPU arch | Cards |
|---|---|---|
| RDNA 4 | `rdna4` | RX 9070 XT / 9070 / 9070 GRE, RX 9060 XT / 9060 |
| RDNA 3 | `rdna3` | RX 7900 XTX / 7900 XT / 7900 GRE, RX 7800 XT, RX 7700 XT, RX 7600 XT / 7600 |
| RDNA 2 | `rdna2` | RX 6950 XT / 6900 XT / 6800 XT / 6800, RX 6750 XT / 6700 XT / 6650 XT / 6600 XT / 6600 |
| RDNA 1 *(planned)* | `rdna1` | RX 5700 XT / 5700 / 5600 XT / 5500 XT |

### Intel

| Architecture | GPU arch | Cards |
|---|---|---|
| Arc Battlemage | `arc_battlemage` | Arc B580 / B570 |
| Arc Alchemist | `arc_alchemist` | Arc A770 / A750 / A580 / A380 / A310 |

### Apple Silicon

| Architecture | GPU arch | Chips |
|---|---|---|
| Metal | `metal` | M1 / M1 Pro / Max / Ultra, M2 / M2 Pro / Max / Ultra, M3 / M3 Pro / Max, M4 / M4 Pro / Max |

Not sure which architecture you have? Run `spite dispatch` — it detects and prints it.

---

## Configuration

Copy `spite.toml` from the repo root and edit it. The main settings:

```toml
[server]
host        = "127.0.0.1"
port        = 8080
max_concurrent = 4

[model]
path        = "models/model.gguf"

[inference]
n_ctx       = 4096    # context length
n_threads   = 4       # CPU fallback threads

[sampling]
temperature = 1.0
top_p       = 0.95
```

Pass it to the server: `spite-server --config spite.toml`

---

## Optional: LoRA adapters

Fine-tuned adapters in GGUF format load alongside the base model. Useful for
domain specialization (code, medical, legal) without swapping the base weights.

```bash
spite-server --model base.gguf --lora my_adapter.gguf
```

---

## Benchmarking

```bash
# Show which kernel is active for each operation on your card
./target/release/spite dispatch -m ~/models/your.gguf --card RTX_4090

# Measure end-to-end model throughput
./target/release/spite-bench --model ~/models/your.gguf
```

`spite dispatch` prints the resolved kernel for each operation:

```
model arch   : llama4
card         : rtx_4090 (24 GiB)
gpu arch     : sm_89
kernels      :
  rms_norm       → sm_89/kernels/llama/llama4/sm_89
  attention      → generic/kernels/generic/generic
  ffn            → sm_89/kernels/llama/llama4/sm_89
  layer          → generic/kernels/generic/generic
  spec_verify    → generic/kernels/generic/generic
  prefill        → generic/kernels/generic/generic
```

Entries resolved to `generic/` are running the fallback — a custom kernel
for that op and GPU would be faster.

---

## Contributing a kernel

This is the heart of the project. You have an RTX 3080. You know it better
than anyone who doesn't own one. A kernel you write for it will be faster
than any generic kernel we could ship.

You don't need to understand the scheduler, the tokenizer, the server, or
anything else. You need:
- Your GPU
- One operation to implement (attention, FFN, or rms_norm)
- The template in `kernels/llama/llama4/sm_89/KERNEL_TEMPLATE.cu`

**The steps:**

```bash
# 1. Find what's slow on your card
spite dispatch -m your.gguf --card RTX_5090

# 2. Copy the template for your card
cp kernels/llama/llama4/sm_89/KERNEL_TEMPLATE.cu \
   kernels/llama/llama4/sm_120/attention.cu

# 3. Implement the op (the template has comments for each section)

# 4. Build the kernels for your card
cmake -B build -DSPITE_MODELS="llama/llama4" -DSPITE_GPU_ARCHS="RTX_5090" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 5. Verify correctness — must pass before PR
python3 tools/verify/verify.py \
  build/kernels/llama/llama4/sm_120/libkernel_llama_llama4_sm_120.so

# 6. Benchmark and save the output
cargo run --release -p spite-bench -- --model your.gguf \
  > kernels/llama/llama4/sm_120/attention.bench

# 7. Open a PR titled:  kernel: llama/llama4/sm_120 attention
```

You only touch the `kernels/` directory. Nothing else breaks when you add a
new file. The dispatcher picks up your kernel automatically.

Full guide: [docs/contributing/CONTRIBUTING.md](docs/contributing/CONTRIBUTING.md)

GPU-specific notes (tile sizes, WMMA shapes, memory layout):
[docs/gpu_guides/](docs/gpu_guides/)

---

## Supported models

| Model               | Status   |
|---------------------|----------|
| llama/llama4        | template |
| deepseek/v4         | template |
| qwen/qwen3_5        | template |
| qwen/qwen4          | template |
| mistral/mistral4    | template |
| gemma/gemma4        | template |
| glm/glm5            | template |
| glm/glm_dsa         | template |
| minimax/m3          | template |
| kimi/k3             | template |
| eagle/eagle3        | template |
| mellum/base         | template |

"Template" means the model layout and loader are wired up; kernel contributions
welcome. Running any of these on the generic fallback works today.

---

## Architecture overview

```
spite run / spite-server
       │
       ▼
  spite-loader          reads .gguf weights from disk (mmap, no copy)
       │
  spite-dispatch        picks the right kernel .so for model × GPU at runtime
       │
  spite-executor        runs the forward pass: GPU layers → CPU fallback
       │
  spite-scheduler       batches concurrent requests into one forward pass
       │
  spite-server          OpenAI-compatible HTTP API (axum)
```

Every component is an independent crate. You can use the scheduler without
the server, the loader without the executor, or just the ABI types for kernel
development. See [docs/architecture.md](docs/architecture.md) for the full
dependency graph and data flow.

---

## Design principles

- **GGUF only.** One weight format. No conversion step, no format zoo.
- **Consumer hardware first.** Every decision optimizes for the RTX 3060 and
  RX 7800 XT before it optimizes for the A100.
- **Your card, your kernel.** Tuned kernels live in `kernels/<family>/<model>/<gpu_arch>/`.
  Adding yours doesn't require touching anything else.
- **Fallback always works.** No kernel for your GPU? The generic CPU fallback runs.
  It's slower, not broken.
- **No cloud, no telemetry.** spite doesn't phone home. It runs on your hardware,
  reads your files, and that's it.
- **Replace anything.** Sampler, tokenizer, KV cache, offload policy — every
  subsystem is a plugin registry. Swap out any piece without forking the project.

---

## Roadmap

- [ ] Flash attention kernels (sm_86, sm_89, rdna3)
- [ ] Speculative decoding (draft model → main model verification)
- [ ] Prefix caching (shared prompt KV cache across requests)
- [ ] Vision models (LLaVA, InternVL, Qwen-VL)
- [ ] Quantization tools (`spite quantize` — convert f16 → Q4_K_M locally)
- [ ] Windows support

---

## License

Apache 2.0. See [LICENSE](LICENSE).

# spite

**One Engine. Your Model. Your Card.**

Most AI inference tools are built around one GPU brand, one cloud, or one company's
model. spite is built around *you*. You pick the model. You pick the hardware. The
engine adapts — not the other way around.

- **One Engine** — a single Rust host that loads any GGUF model and runs it.
  No cloud required, no subscription, no data leaving your machine.
- **Your Model** — any GGUF file works. Download a model once, run it forever.
  Llama 3, Mistral, Phi-3, Qwen, DeepSeek — if it's a `.gguf`, spite loads it.
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

## Quick start

```bash
# Build
cargo build --release

# Run a model
./target/release/spite run \
  --model ~/models/llama-3-8b-instruct.Q4_K_M.gguf \
  --prompt "What is the capital of France?"

# Start an API server (OpenAI-compatible)
./target/release/spite-server \
  --model ~/models/llama-3-8b-instruct.Q4_K_M.gguf \
  --port 8080
```

Then point any OpenAI-compatible app at `http://localhost:8080`.

---

## Getting a model

Models are distributed as `.gguf` files. The most common source is
[Hugging Face](https://huggingface.co/models?library=gguf). Search for the
model you want, filter by GGUF, and download a quantized version that fits
your GPU's VRAM.

| VRAM available | Recommended quant  | Example                         |
|----------------|--------------------|---------------------------------|
| 4 GB           | Q4_K_M             | 7B model fits comfortably       |
| 8 GB           | Q5_K_M or Q6_K     | 13B model or larger 7B          |
| 12 GB          | Q8_0 or f16        | 13B at high quality             |
| 24 GB+         | f16                | 70B at Q4 or 34B at f16         |

If you don't specify a quantization, spite uses your full GPU memory as
efficiently as possible — automatically stepping down from f16 to Q8 to Q5
to Q4 as it fills.

---

## Building GPU kernels

The Rust host runs without GPU kernels (using the generic CPU fallback), but
for full speed you'll want to compile the kernels for your GPU.

```bash
# Find your GPU architecture first — the benchmark tool shows it:
./target/release/spite benchmark --model ~/models/your.gguf

# Build kernels for your card
cmake -B build \
  -DSPITE_MODELS="llama3"    \
  -DSPITE_GPU_ARCHS="sm_89"  \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Then run with kernels
./target/release/spite run \
  --model ~/models/your.gguf \
  --kernels-dir build/kernels \
  --prompt "Hello"
```

**You don't need to build kernels for every model or every GPU.** Build only
the combination you actually use. The fallback covers everything else.

---

## GPU architecture reference

| Architecture  | Cards                               | Build flag         |
|---------------|-------------------------------------|--------------------|
| sm_89         | RTX 4060–4090, RTX 4000 Ada         | `sm_89`            |
| sm_86         | RTX 3060–3090, RTX A series         | `sm_86`            |
| sm_75         | RTX 2060–2080 Ti, GTX 1660 Ti+      | `sm_75`            |
| rdna3         | RX 7600–7900 XTX                    | `rdna3`            |
| rdna2         | RX 6600–6950 XT                     | `rdna2`            |
| arc_alchemist | Intel Arc A series                  | `arc_alchemist`    |
| metal         | Apple M1 / M2 / M3 / M4             | `metal`            |

Not sure which you have? Run `spite benchmark` — it detects and prints it.

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
./target/release/spite benchmark --model ~/models/your.gguf
```

Output shows which kernel is active for each operation and how fast it is:

```
[dispatch] model: llama3  gpu: sm_86
  rms_norm  → kernels/generic/generic     12.3 µs
  attention → kernels/generic/generic    841.2 µs  ← opportunity
  ffn       → kernels/llama3/sm_86       192.1 µs
```

Lines marked `generic/generic` are running the fallback — a custom kernel
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
- The template in `kernels/llama3/sm_89/KERNEL_TEMPLATE.cu`

**The steps:**

```bash
# 1. Find what's slow
spite benchmark --model your.gguf

# 2. Copy the template for your card
cp kernels/llama3/sm_89/KERNEL_TEMPLATE.cu kernels/llama3/sm_86/attention.cu

# 3. Implement the op (the template has comments for each section)

# 4. Verify correctness — must pass before PR
spite verify kernels/llama3/sm_86/attention.cu

# 5. Benchmark and save the output
spite bench kernels/llama3/sm_86/attention.cu > kernels/llama3/sm_86/attention.bench

# 6. Open a PR titled:  kernel: llama3/sm_86 attention
```

You only touch the `kernels/` directory. Nothing else breaks when you add a
new file. The dispatcher picks up your kernel automatically.

Full guide: [docs/contributing/CONTRIBUTING.md](docs/contributing/CONTRIBUTING.md)

GPU-specific notes (tile sizes, WMMA shapes, memory layout):
[docs/gpu_guides/](docs/gpu_guides/)

---

## Supported models

| Model     | Status   |
|-----------|----------|
| llama3    | template |
| mistral   | template |
| phi3      | template |

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
development.

---

## Design principles

- **GGUF only.** One weight format. No conversion step, no format zoo.
- **Consumer hardware first.** Every decision optimizes for the RTX 3060 and
  RX 7800 XT before it optimizes for the A100.
- **Your card, your kernel.** Tuned kernels live in `kernels/<model>/<gpu_arch>/`.
  Adding yours doesn't require touching anything else.
- **Fallback always works.** No kernel for your GPU? The generic CPU fallback runs.
  It's slower, not broken.
- **No cloud, no telemetry.** spite doesn't phone home. It runs on your hardware,
  reads your files, and that's it.

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

MIT OR Apache-2.0 — your choice.

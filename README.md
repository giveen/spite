# spite

A hyper-modular LLM inference engine for consumer hardware.

Each GPU architecture gets its own kernel folder. Each model gets its own
model folder. You compile only what you need. Anyone can contribute a
faster kernel for their specific card without touching anything else.

## Philosophy

- **GGUF only.** One weight format, simple loader, no conversion step.
- **Consumer hardware first.** RTX 3060, RX 7800 XT, Intel Arc, Apple M2.
  Not data centers.
- **Your card, your kernel.** If you have an RTX 3070, you can write the
  fastest RTX 3070 kernel possible and ship it here.
- **Fallback always works.** Missing a kernel for your GPU? The generic
  fallback runs. It's slower, not broken.

## Build

Two separate build steps: Rust host first, then C++ kernels.

```bash
# 1. Rust host + CLI
cargo build --release

# 2. C++23 kernels (only your GPU × your models)
cmake -B build \
  -DSPITE_MODELS="llama3"      \
  -DSPITE_GPU_ARCHS="sm_89"    \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Kernels land in `build/kernels/<model>/<arch>/`. The CLI finds them at runtime.

## Run

```bash
./target/release/spite run \
  --model path/to/model.gguf \
  --kernels-dir build/kernels \
  --prompt "Hello, world"
```

## Benchmark

```bash
./build/spite benchmark --model path/to/model.gguf
```

Shows which kernel is active for each op and how fast it is.

## Contributing a kernel

See [docs/contributing/CONTRIBUTING.md](docs/contributing/CONTRIBUTING.md).

Short version: copy the template, implement one op, run `spite verify` and
`spite bench`, paste the bench output in your PR.

## Supported GPUs

| Architecture | Folder       | Cards                              | Status      |
|--------------|--------------|------------------------------------|-------------|
| sm_89        | sm_89/       | RTX 4060–4090, RTX 4000 Ada        | template    |
| sm_86        | sm_86/       | RTX 3060–3090, RTX A series        | template    |
| sm_75        | sm_75/       | RTX 2060–2080 Ti, GTX 1660 Ti+     | template    |
| rdna3        | rdna3/       | RX 7600–7900 XTX                   | template    |
| rdna2        | rdna2/       | RX 6600–6950 XT                    | template    |
| arc_alchemist| arc_alchemist/| Intel Arc A series                | template    |
| metal        | metal/       | Apple M1/M2/M3/M4                  | template    |

All architectures fall back to `kernels/generic/generic/` until someone
contributes a tuned kernel.

## Supported models

| Model     | Folder   | Status   |
|-----------|----------|----------|
| llama3    | llama3/  | template |
| mistral   | mistral/ | template |
| phi3      | phi3/    | template |

## Directory structure

```
core/          ABI contract (abi.h), quant block layouts (quant.h)
dispatch/      Runtime kernel selection
loader/        GGUF file loader
kernels/
  <model>/
    <gpu_arch>/   Your kernel lives here
    generic/      Fallback — always correct, unoptimized
  generic/        Cross-model generic ops
tools/
  benchmark/    Measures op latency, shows which kernel is active
  verify/       Checks kernel output against reference
  tune/         Sweeps tile configs to find the fastest for your GPU
docs/
  gpu_guides/   Per-architecture notes: tensor cores, tile sizes, gotchas
  contributing/ How to write and submit a kernel
```

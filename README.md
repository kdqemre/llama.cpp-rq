# llama.cpp - Rotated Quantization (RQ) Edition

*This repository is a specialized fork of the phenomenal [llama.cpp](https://github.com/ggml-org/llama.cpp) project by **Georgi Gerganov** and the ggml community. All core engine and architecture credits belong to the original developers.*

Welcome to the **Rotated Quantization (RQ)** edition of `llama.cpp`. This repository introduces Rotated Quant formats — **RQ2 (2-bit), RQ3 (3-bit), RQ4 (4-bit)** — utilizing the **Walsh-Hadamard Transform (WHT)**.

My goal is to push LLM quantization down to extreme low-bit levels (2 / 3 / 4-bit) without the catastrophic reasoning degradation seen in standard quantization, while **maintaining native `llama.cpp` inference speeds** (same `t/s`, sometimes faster than the reference K-quants).

---

## 🧠 Why Rotation? (The Math Behind WHT)

In standard quantization (like `Q4_0` or `Q4_K`), weights are grouped into blocks. The quantization scale
$\Delta$ is determined by the absolute maximum value (outlier) in that block:

$$ \Delta = \frac{\max(|x|)}{2^{b-1} - 1} $$

### The Outlier Problem

Imagine a block of 32 weights where 31 values are `1.0` and one outlier is `45.0`. In a standard 4-bit
format, the scale becomes
$\Delta \approx 6.43$. The 31 small weights are divided by 6.43 (yielding ≈ 0.15) and rounded to **zero**.
The block loses all its fine-grained information, and the reconstruction error concentrates entirely on the inlier mass.

### The RQ Solution

Before quantizing, we multiply the weight-block vector $x$ by a Walsh-Hadamard matrix
$H_N$ ($w = H_N x$, then normalize by $1/\sqrt{N}$). The WHT acts as a mathematical "blender," distributing
the magnitude of the outlier **evenly across all elements** — because every Hadamard row is a balanced
{±1} combination, no single lane carries the whole outlier.

In our example, the WHT shrinks the value range from `[1, 45]` to `[-7.95, 13.44]`. The new scale becomes
$\Delta \approx 1.92$. **Every single lane now gets a real representable level**, reducing the Normalized
Mean Square Error (NMSE) by **1–2 orders of magnitude** at the *exact same bit rate*.

Since $H_N$ is orthogonal ($H_N^\top H_N = N \cdot I$), the inverse `H_N^\top y / \sqrt{N}` recovers the
natural-domain weights losslessly up to the rotated-domain quantization error — and because that error is
uniform across lanes, the natural-domain reconstruction is uniformly clean instead of being dominated by a
few crushed buckets.

### Fixed diagonal sign pattern (no calibration)

The WHT alone is a fixed, calibration-free rotation; multiplying by a `{±1}^{32}` diagonal before it is still
a Hadamard-style rotation, but the *choice of signs* shifts which outliers land in which lanes — and on real
model weights the resulting round-trip PPL **varies by many points across sign patterns**. We therefore run
a small, fast global 32-bit coordinate-ascent sign search on a proxy model and **bake the winner directly
into the code as the default** (no env, no calibration at quantize- or serve-time).

The RQ2 winner (`-+---+-+-+-++-++++----+++---+++-`) dropped proxy PPL 74.58 → 53.44 on Qwen3.5-0.8B in a
single search pass — empirically the same insight behind QuaRot / SpinQuant / FlatQuant, specialized here to
llama.cpp's block layout.

---

## 🚀 The RQ Ladder & Inference Performance

Our implementation ensures that **you do not lose tokens/second for better math.** Weights are **pre-rotated
at quantization time** (the GGUF stores the rotated weights, so there is zero per-token weight rotation at
inference). During inference the **activation WHT is fused directly** into the existing kernels
(`quantize_mmq_q8_1` for prefill, the MMVQ dot for decode), and the heavy matrix multiplication runs via
**standard `dp4a` hardware instructions**. There is no separate WHT pass, no correction kernel, no
calibration step, no cuBLAS fallback.

**Reference hardware:** NVIDIA RTX 4090 (sm_89) | **Model:** Qwen3.6-27B (qwen35 hybrid, 65 blocks)
GSM8K-100 accuracy is the intelligence metric (8-shot greedy, `llama-server`); decode speed is the
server's `predicted_per_second` on real prompts.

| Format | BPW (overall) | Block Size | GSM8K-100 (acc) | Decode (t/s) | Notes vs the reference `Q_K` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **RQ2_K_L** | 3.70 | 80 B | **84 / 100** | **36.1** | vs `Q2_K`: **2.5× accuracy** (84 vs 34) **and +14% faster decode** (36.1 vs 31.6 t/s). Bulk is smaller than Q2_K (80 B vs 84 B). |
| **RQ3_K_L** | ~3.5 | 112 B | **83 / 100** | 27.0 | vs `Q3_K_M`: **+14 GSM8K points** (83 vs 69), same decode class. |
| **RQ4_K_L** (combo6_q5) | 4.93 | 144 B | **64 / 100** | 34.1 | vs `Q5_K_M`: **+10 points** (64 vs 54) **while saving 2.7 GB VRAM** (16.8 vs 19.5 GB). |

### `K_L` elevation tier (mixed precision, automatic)

Each `_K_L` preset is **not** a uniform quant — it defaults to a **mixed-precision recipe** that keeps the
format's raw low bit-rate on the insensitive bulk and **elevates** the few sensitive tensor categories one
llama.cpp tier up (e.g. for RQ2_K_L on Qwen3.6-27B: FFN down → Q5_K, attn qkv / v → Q5_K, token embedding →
Q3_K, attn output → Q4_K, **bulk → RQ2**). The bulk ends up *cheaper* than the nearest standard quant
(RQ2's 80 B block < Q2_K's 84 B → less memory traffic → faster decode), and the ~+1.7 GB over Q2_K comes
entirely from that elevation — which is exactly what doubles GSM8K accuracy (84/100 vs 34/100). Every
category is overridable per-tensor with `--tensor-type` (see §Advanced).

Supports **Speculative Decoding (draft-MTP)** out of the box (51.3 t/s at 72% draft acceptance on the
27B, answers bit-identical to plain — speculative verify). Verified on dense (Qwen3) **and** MoE
(`qwen3moe` 4×0.6B with fused `gate_up_exps`) — the rotation sits on the activation's `n_embd` dim and is
orthogonal to expert selection.

---

## 🛠️ Building the Project

Build the project exactly as you would with standard `llama.cpp`. We recommend using `-j 8` for stability
(unbounded parallelism can thermal-trip consumer GPUs).

```bash
# Clone the repository
git clone https://github.com/kdqemre/llama.cpp-rq.git
cd llama.cpp-rq

# Build with CUDA support (RTX 40-series example)
cmake -S . -B build_cuda \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build_cuda \
      --target llama-quantize llama-server llama-cli llama-perplexity -j 8
```

> Reproducibility tip — never pipe `llama-quantize` output through `head`/`grep`: SIGPIPE can abort the
> writer mid-stream and corrupt the GGUF. Redirect output to a log file (`> /tmp/q.log 2>&1`).

---

## 📦 Usage: Standard Quantization

We provide highly optimized, pre-configured recipes (the `_K_L` variants) that balance the model
architecture automatically — keeping the raw low-bit format on the bulk and elevating the sensitive
categories.

```bash
# 2-bit tier  -> RQ2_K_L  (2.5x Q2_K GSM8K + faster decode)
./build_cuda/bin/llama-quantize Model-BF16.gguf Model-RQ2_K_L.gguf RQ2_K_L

# 3-bit tier  -> RQ3_K_L  (+14 GSM8K points over Q3_K_M)
./build_cuda/bin/llama-quantize Model-BF16.gguf Model-RQ3_K_L.gguf RQ3_K_L

# 4-bit tier  -> RQ4_K_L  (beats Q5_K_M on accuracy + smaller)
./build_cuda/bin/llama-quantize Model-BF16.gguf Model-RQ4_K_L.gguf RQ4_K_L
```

### Serve

```bash
./build_cuda/bin/llama-server -m Model-RQ2_K_L.gguf -c 8192 -ngl 999 --temp 0.0 -fa on

# + speculative decode (the model must contain an MTP draft head; Qwen3.6-27B does):
#   add  --spec-type draft-mtp --spec-draft-n-max 3
```

### Self-check the kernels (optional)

Four validation harnesses ship in the repo and compare the CUDA kernels against the CPU reference:

```bash
cmake --build build_cuda --target test-rq2-cpu test-rq2-cuda test-rq3-k-l-cuda test-rq4-cuda -j 8
./build_cuda/bin/test-rq2-cpu   ./build_cuda/bin/test-rq2-cuda
./build_cuda/bin/test-rq3-k-l-cuda   ./build_cuda/bin/test-rq4-cuda
```

---

## 🔬 Usage: Advanced Modular Quantization

This fork exposes a powerful **repeatable** `--tensor-type <regex>=<ggml_type>` argument. Instead of
locking the whole model into one format, you can selectively quantize specific layers (Attention, FFN,
Embeddings) using regular expressions. The type token is the **lowercase `ggml_type_name`**, i.e.
`rq2` / `rq3` / `rq4` / `q4_K` / `q5_K` / `q6_K` / … — these override the base recipe per-tensor.

**Example — a hybrid RQ2 + RQ3 + RQ4 model (~3.65 bpw total).** Here we use `RQ2_K_L` as the lightweight
base (bulk = RQ2 + its auto-elevation), then ** forcibly override** the FFN gate/up layers to `RQ3` and the
most sensitive attention/FFN tensors to `RQ4`:

```bash
./build_cuda/bin/llama-quantize \
  --tensor-type 'blk\.\d+\.ffn_(gate|up)\.weight=rq3' \
  --tensor-type 'blk\.\d+\.(attn_qkv|attn_v|ffn_down)\.weight=rq4' \
  Model-BF16.gguf \
  Model-Mixed.gguf \
  RQ2_K_L
```

The quantizer logs each applied override (`applying manual override: rq2 -> rq4`), and you can verify the
resulting multi-type GGUF with the bundled `gguf-py`:

```python
import sys; sys.path.insert(0, "gguf-py")
import gguf, collections
r = gguf.GGUFReader("Model-Mixed.gguf")
print(collections.Counter(t.tensor_type.name for t in r.tensors))
# Counter({'RQ2': 84, 'RQ3': 48, 'RQ4': 48, 'Q4_K': 6, 'Q3_K': 1, 'F32': 133})
```

Other useful patterns:

```bash
# Force RQ2 everywhere — the pure 2.50 bpw baseline (no K_L elevation):
./build_cuda/bin/llama-quantize --tensor-type '.*\.weight=rq2' Model-BF16.gguf Model-RQ2-modular.gguf RQ2_K_L

# Recreate the 27B RQ4 "combo6_q5" winner (Q5_K elevation on top of RQ4 bulk): 16.8 GB, GSM8K 64
./build_cuda/bin/llama-quantize \
  --tensor-type 'blk\.\d+\.(attn_qkv|attn_v|attn_output|ffn_down)\.weight=q5_K' \
  --tensor-type '(output|token_embd)\.weight=q5_K' \
  Model-BF16.gguf Model-RQ4-combo6_q5.gguf RQ4_K_L
```

If you have too many rules, you can also pass them in `--tensor-type-file rules.txt` (one `regex=type` per
line). Overrides take precedence over the base recipe per-tensor.

---

## ⚙️ Compatibility & Status

- **Type slots (zero collisions):** we use `GGML_TYPE_RQ4 = 47`, `GGML_TYPE_RQ3 = 48`,
  `GGML_TYPE_RQ2 = 49` (`GGML_TYPE_COUNT = 50`), paired with the matching
  `LLAMA_FTYPE_MOSTLY_RQ{4,3,2}_K_L = 47/48/49` ftypes — upstream hasn't added new `GGML_TYPE_*` enums, so
  these are collision-free with standard GGML types. Python-side (`gguf-py/gguf/constants.py`) mirrors the
  type ids and the correct block-size entries, so Python GGUF tooling reads RQ files.
- **CPU offload:** Full `-ngl` support is included. RQ kernels are registered in the CPU traits table
  (`ggml_vec_dot_rq{2,3,4}_f32`), so you can split RQ models between CPU RAM and GPU VRAM with **no
  correctness loss** — a 27B RQ2_K_L with `-ngl 33` puts ~7 GB on the 4090 and ~6.6 GB in host RAM and
  still answers correctly (decode is limited by the CPU half). Full-GPU (`-ngl 999`) and full-CPU
  (`-ngl 0`) both work.
- **Architecture support:** Fully tested with **dense** models (Qwen3.5/3.6), **Mixture-of-Experts**
  (`qwen3moe` with fused `gate_up_exps`), hybrid Mamba/Gated-Delta-Net layers, and **Speculative /
  draft-MTP** — all produce correct, content-identical output to the corresponding standard `Q_K`.
- **Features preserved:** CUDA graphs, flash-attention (`-fa on`), the standard `--tensor-type` /
  `--tensor-type-file` overrides, `llama-bench`, full graph-level fault isolation — everything upstream.
- **Self-checks:** `test-rq2-cpu.cpp`, `test-rq2-cuda.cpp`, `test-rq3-k-l-cuda.cpp`, `test-rq4-cuda.cpp`
  (two-backend graph compare, get_rows bit-exact, round-trip NMSE gates).

### Hybrid-arch note

For hybrid Mamba/Gated-Delta-Net models, the **fused chunked-GDN** op can't span a CPU/GPU device split —
when you partial-offload with `-ngl < n_layers` the server logs
`fused Gated Delta Net (chunked) not supported, set to disabled` and falls back to the non-fused GDN path.
Output stays correct; the caveat is performance-only. Full-GPU and full-CPU runs are unaffected.

---

## 📄 License & Acknowledgements

This project is **MIT-licensed**, inherited from [llama.cpp](https://github.com/ggml-org/llama.cpp). Huge
thanks to **Georgi Gerganov** and every contributor to `ggml-org/llama.cpp` — this fork is a small delta on
a very large and excellent foundation. The rotation-before-quantization principle it instantiates is also
indebted to the published research on calibration-free orthogonal rotation for LLM quantization
(QuaRot, SpinQuant, FlatQuant).

`llama.cpp-rq` is an independent fork and is **not affiliated with or endorsed by** the
ggml-org/llama.cpp project.

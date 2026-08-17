# RQ — Rotated Quantization for llama.cpp

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml 0.19.0 base, commit `885c5bbe`) adding a family of **WHT-rotated K-quants** — RQ2, RQ3, RQ4 at 2/3/4 bits — as univer[...]

| type | ggml ftype | block layout | nominal | measured @ Qwen3.8-27B |
|---|---:|---|---:|---:|
| `RQ2_K_L` | 49 | 80 B / 256 weights | 2.50 bpw | **3.70 bpw, 12.6 GB** |
| `RQ3_K_L` | 48 | 112 B / 256 weights | 3.50 bpw | **4.33 bpw, 14.8 GB** |
| `RQ4_K_L` | 47 | 144 B / 256 weights (byte-identical to `block_q4_K`) | 4.50 bpw | **5.25 bpw, 17.9 GB** |

One idea powers all three: **before quantizing, every 32-weight sub-block is rotated by a signed Walsh–Hadamard transform.** The rotation spreads outlier energy evenly across the sub-block, so t[...]

---

## Why does this repository exist?

LLM weight matrices contain **outliers** — a few weights 10–100× larger than the rest. Block quantization divides its codebook by the block's range, so one outlier inflates the step size for [...]

### Standard block quantization (any bit width $b$)

For a block $\mathbf w \in \mathbb R^B$:

$$s = \frac{w_{\max} - w_{\min}}{2^b - 1}, \\\qquad q_i = \mathrm{round}\!\left(\frac{w_i - w_{\min}}{s}\right), \\\qquad \hat w_i = w_{\min} + q_i\, s$$

The step $s$ is set by the **range**, not by where the values actually live.

**Toy example ($b=4$, block of 4),** $\mathbf w = [1,\ 2,\ 3,\ 100]$:

$$s = \tfrac{100-1}{15} = 6.60 \;\Rightarrow\; \mathbf q = [0,\ 0,\ 0,\ 15] \;\Rightarrow\; \hat{\mathbf w} = [1,\ 1,\ 1,\ 100]$$

| element | 1 | 2 | 3 | 100 |
|---|---:|---:|---:|---:|
| quantized | 1 | **1** | **1** | 100 |
| relative error | 0% | **−50%** | **−66.7%** | 0% |

The three ordinary values collapse onto a single code; 14 of 16 levels are wasted on the outlier's range. At $b=2$ the same block is worse still: $s=33$, codes $[0,0,0,3]$ — everything but the o[...]

### Rotated quantization (this repo)

The Walsh–Hadamard matrix (Sylvester construction) is orthogonal up to scale:

$$H_2 = \begin{pmatrix} 1 & 1 \\\ 1 & -1 \end{pmatrix}, \\\qquad H_{2n} = H_n \otimes H_2, \\\qquad H_n H_n^\top = nI$$

RQ rotates every 32-weight sub-block with a **signed** WHT — a per-lane sign diagonal $D = \mathrm{diag}(\pm 1)$ followed by the fast butterfly:

$$\mathbf x' = \tfrac{1}{\sqrt{32}}\, H_{32}\, D\, \mathbf x, \\\qquad \lVert \mathbf x' \rVert = \lVert \mathbf x \rVert$$

The transform preserves the L2 norm and **dilutes spikes**. Same toy block (shown with $H_4$ for readability):

$$\tfrac{1}{2}H_4\,[1,2,3,100] = [53,\ -49,\ -50,\ 48]$$

The spike's energy is now spread over all four coordinates, which all quantize at comparable magnitude — the codebook is used evenly. Sparse-outlier case ($b=4$, block of 8), $\mathbf w = [\,0.5[...]

| | codes used | step | L2 error | ordinary values |
|---|---:|---:|---:|---:|
| standard | {0, 1, 15} — collapse | 2.63 | 2.02 | 1.0→0.5, 2.0→3.13 (−50…+57%) |
| **rotated** | {−6,−5,5,6,7} — even | 2.40 | **1.86** | errors ≤ ~1.2, uniform |

The full pipeline, per sub-block at any bit width:

$$\hat{\mathbf w} = \tfrac{1}{\sqrt{32}} H_{32}\; \mathcal Q^{-1}\!\big(\mathcal Q_b(\tfrac{1}{\sqrt{32}} H_{32} D\, \mathbf w)\big)$$

At inference the standard K-quant scale/index machinery applies unchanged; the rotation itself is paid **once per activation token** (fused into the activation quantizer), not once per weight — [...]

A toy block can't show the whole picture — the payoff is statistical. Across a real matrix, outlier coordinates are sparse; rotation guarantees every sub-block absorbs an equal slice of every ou[...]

> **Qwen3.6-27B, 0-shot, MTP:** RQ2_K_L **92** vs Q2_K **64** (+28) · RQ3_K_L-rqmod **97** vs Q3_K_M 93 · RQ4_K_L-rqmod 88 vs Q4_K_M 89
> **Qwen3.8-27B, 8-shot, MTP-off:** RQ2_K_L 93 vs Q2_K 91 · RQ2_K_L-rqmod **98** · 3-bit tier: RQ3_K_L 97 = Q3_K_M 97 · 4-bit tier: RQ4_K_L 97 = Q4_K_M 97

**Honest scoping:** rotation helps where bits are scarce (2/3/4). At 5 bits the error-spreading that makes rotation great at low bits starts hurting exact multi-step arithmetic (a few natural-doma[...]

---

## What can you do with this repo?

### Quantize — standard recipes

```bash
llama-quantize model-f16.gguf model-rq2.gguf RQ2_K_L 8
llama-quantize model-f16.gguf model-rq3.gguf RQ3_K_L 8
llama-quantize model-f16.gguf model-rq4.gguf RQ4_K_L 8
```

Each `_K_L` recipe mirrors the matching K-quant Large mixture — the bulk (FFN gate/up, SSM paths, small projections) gets the rotated low-bit type; sensitive tensors (attention projections, FFN-[...]

### Quantize — modular recipes (`--tensor-type`)

RQ types are **universal tensor types**: mix them per tensor group with regex overrides. Keep a 2-bit bulk, elevate only what matters:

```bash
# RQ2_K_L-rqmod — 98/100 GSM8K on Qwen3.8-27B at 3.70 bpw
llama-quantize --tensor-type 'token_embd\\.weight=rq3' \\
               --tensor-type 'blk\\.[0-9]+\\.attn_output\\.weight=rq4' \\
               model-f16.gguf model-rq2rqmod.gguf RQ2_K_L 8

# RQ3_K_L-rqmod — 98/100 at 4.28 bpw
llama-quantize --tensor-type 'token_embd\\.weight=rq4' \\
               --tensor-type 'blk\\.[0-9]+\\.attn_output\\.weight=rq4' \\
               model-f16.gguf model-rq3rqmod.gguf RQ3_K_L 8

# RQ4_K_L-rqmod — bulk RQ4, elevate QKV/V/FFN-down to Q5_K
llama-quantize --tensor-type 'blk\\.[0-9]+\\.(attn_qkv|attn_v|ffn_down)\\.weight=q5_K' \\
               --tensor-type 'blk\\.[0-9]+\\.attn_output\\.weight=rq4' \\
               model-f16.gguf model-rq4rqmod.gguf RQ4_K_L 8
```

> **Gotcha:** `--tensor-type` options must appear **before** the model path — the arg parser stops consuming options at the first positional, so overrides placed after the paths are silently ig[...]

### Run it

```bash
# server
llama-server -m model-rq3.gguf -ngl 99 -c 8192 -t 8 -fa on \\
  --jinja --reasoning off --temp 0.0 \\
  --cache-type-k q8_0 --cache-type-v q8_0

# MTP (models with blk.64.nextn heads — Qwen3.6/3.8 hybrids): verified 1.95× decode
llama-server -m model-rq4.gguf ... --spec-type draft-mtp --spec-draft-n-max 3

# one-shot CLI (non-interactive)
llama-cli -m model-rq3.gguf -ngl 99 -t 8 -n 80 --temp 0.0 \\
  -p "Q: ...\nA:" -no-cnv --single-turn
```

### What's inside (speed & quality work)

- **Optimized rotation-sign search** — the $D$ diagonal is not random: offline sweeps over sign patterns (70+ candidates) move wikitext PPL by up to **1.4 points**; per-category diagonals add u[...]
- **Fused CUDA kernels for all three types** — the WHT is fused into the activation quantizer (`prep_act`, once per token); the matmuls are plain dp4a/DP4A MMA over K-quant-layout blocks (RQ4's[...]
- **MoE verified**, MTP draft heads quantize cleanly (84.9% acceptance on RQ4_K_L), fused GLU (gate+up+SwigLU) supported, plain-greedy decode 26–35 t/s at 27B on a single RTX 4090.

### Evidence (Qwen3.8-27B, 8-shot GSM8K-100, MTP-off, one harness)

| scheme | bpw | size | GSM8K-100 | decode t/s |
|---|---:|---:|---:|---:|
| RQ2_K_L | 3.70 | 12.0 GB | 93 | 34.7 |
| **RQ2_K_L-rqmod** | 3.70 | 12.0 GB | **98** | 32.3 |
| Q2_K | 3.18 | 10.1 GB | 91 | 28.2 |
| RQ3_K_L | 4.33 | 13.8 GB | 97 | 25.8 |
| RQ3_K_L-rqmod | 4.28 | 13.6 GB | 98 | 23.7 |
| Q3_K_M | 3.95 | 12.6 GB | 97 | 26.1 |
| RQ4_K_L | 5.25 | 16.7 GB | 97 | 26.6 |
| RQ4_K_L-rqmod | 4.91 | 15.6 GB | 97 | 29.8 |
| Q4_K_M | 4.92 | 15.6 GB | 97 | 31.7 |
| Q5_K_M | 5.72 | 18.2 GB | 98 | 29.2 |

Harness: canonical 8-shot CoT primer, plain greedy (temp 0, seed 0), ctx 8192, n_predict 4096, KV q8_0, `stop=["####"]`, RTX 4090.

### Build & tests

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \\
      -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FA=ON \\
      -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build -j 8
ctest --test-dir build -R 'rq2|rq3|rq4'   # NMSE + kernel parity suites
```

---

Fork base: `ggml-org/llama.cpp` @ `885c5bbe` (ggml 0.19.0). RQ types are **fork-only** — stock llama.cpp refuses these GGUFs; always serve them with a build from this repository.

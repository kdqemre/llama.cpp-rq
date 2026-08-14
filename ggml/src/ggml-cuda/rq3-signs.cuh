// RQ3 sign-pattern parameterization (research sweep).
//
// The RQ3 format applies a per-32-element-subblock randomized Hadamard
// transform whose diagonal sign pattern is a RUNTIME parameter, independent of
// RQ4's, so the RQ3 sign search can be swept without disturbing RQ4:
//
//   RQ3_SIGNS = exactly 32 chars of '+'/'-'  (lane-wise ±1 diagonal)
//
// Default (env unset OR malformed): the rand2 literal (byte-identical to
// RQ3_SIGNS[] in ggml-quants.c), a strong prior — WHT decorrelation is
// format-agnostic. Until the Phase-6 sign search bakes the winner, this
// rand2 default is the self-consistent quantize+inference rotation.
//
// IMPORTANT: the quantizer (llama-quantize) and the inference engine (llama-cli
// / llama-perplexity) are SEPARATE processes. Both must be launched with the
// SAME RQ3_SIGNS so the GGUF's baked rotation matches the runtime activation
// rotation; otherwise dequant of the weights uses a different diagonal than the
// one used at bake time and the result is garbage.
//
// Mechanism: every CUDA TU that consumes the RQ3 rotation includes this
// header, which defines a TU-local `static __constant__` device symbol
// initialized to the rand2 literal (byte-identical default even if the init
// function is never called). Each TU's RQ3 host entry point calls
// `ggml_cuda_rq3_init_signs()` once before its first kernel launch; the
// function parses RQ3_SIGNS, uploads the parsed pattern to that TU's device
// constant via cudaMemcpyToSymbol, and becomes a no-op on subsequent calls.

#pragma once

#include <cstdlib>
#include <cstring>

// Phase-6 global coord-ascent winner (0.8B wikitext-2, 64-chunk proxy). Beats
// the rand2 prior by 3.40 PPL (21.67 -> 18.27) and Q3_K (19.80). Identical
// byte-for-byte to RQ3_SIGNS[] in ggml-quants.c. Used both as the device
// symbol's initializer (byte-identical default) and the env-parse fallback.
#define GGML_CUDA_RQ3_SIGNS_GOLDEN                                    \
    -1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f,                \
    -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, +1.0f,                \
    +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f,                \
    +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, -1.0f

// TU-local device symbol. `static` => one copy per translation unit. Each TU
// that includes this header gets its own, initialized to the winner literal.
static __constant__ float ggml_cuda_rq3_signs_dev[32] = {
    GGML_CUDA_RQ3_SIGNS_GOLDEN,
};

// Parse RQ3_SIGNS into `out` (32 floats, each ±1). Default (unset/malformed):
// the rand2 literal. Centralized so every TU agrees on the format and the
// default.
static inline void ggml_cuda_rq3_signs_parse(float out[32]) {
    static const float golden[32] = { GGML_CUDA_RQ3_SIGNS_GOLDEN };
    const char * env = std::getenv("RQ3_SIGNS");
    bool ok = false;
    if (env) {
        std::size_t n = 0;
        while (env[n] && n <= 32u) ++n;
        if (n == 32u) {
            ok = true;
            for (int i = 0; i < 32; ++i) {
                if      (env[i] == '+') out[i] = +1.0f;
                else if (env[i] == '-') out[i] = -1.0f;
                else { ok = false; break; }
            }
        }
    }
    if (!ok) std::memcpy(out, golden, sizeof(golden));
}
#undef GGML_CUDA_RQ3_SIGNS_GOLDEN

// Idempotent host init: parse RQ3_SIGNS once, upload the 32-float pattern to
// THIS TU's `ggml_cuda_rq3_signs_dev` via cudaMemcpyToSymbol. Call from
// any RQ3 host entry point before the first kernel launch.
static inline void ggml_cuda_rq3_init_signs() {
    static bool inited = false;
    if (inited) return;
    float h[32];
    ggml_cuda_rq3_signs_parse(h);
    cudaMemcpyToSymbol(ggml_cuda_rq3_signs_dev, h, sizeof(h));
    inited = true;
}

// RQ4 sign-pattern parameterization (research sweep).
//
// The RQ4 format applies a per-32-element-subblock randomized Hadamard
// transform whose diagonal sign pattern was historically a compile-time golden-
// ratio literal (RQ4_SIGNS[] in ggml-quants.c, duplicated verbatim in every
// CUDA kernel that consumes the RQ4 rotation). For the sign-pattern sweep
// the pattern is now a RUNTIME parameter:
//
//   RQ4_SIGNS = exactly 32 chars of '+'/'-'  (lane-wise ±1 diagonal)
//               e.g. RQ4_SIGNS="+-+--+--+-+--+--+-+--+--+-+--+--"
//
// Default (env unset OR malformed): the golden-ratio literal, byte-identical to
// the historical compile-time constant => bit-identical GGUFs and bit-identical
// inference when the env var is unset.
//
// IMPORTANT: the quantizer (llama-quantize) and the inference engine (llama-cli
// / llama-perplexity) are SEPARATE processes. Both must be launched with the
// SAME RQ4_SIGNS so the GGUF's baked rotation matches the runtime activation
// rotation; otherwise dequant of the weights uses a different diagonal than the
// one used at bake time and the result is garbage.
//
// Mechanism: every CUDA TU that consumes the RQ4 rotation includes this
// header, which defines a TU-local `static __constant__` device symbol
// initialized to the golden literal (byte-identical default even if the init
// function is never called). Each TU's RQ4 host entry point calls
// `ggml_cuda_rq4_init_signs()` once before its first kernel launch; the
// function parses RQ4_SIGNS, uploads the parsed pattern to that TU's device
// constant via cudaMemcpyToSymbol, and becomes a no-op on subsequent calls.

#pragma once

#include <cstdlib>
#include <cstring>

// Golden-ratio literal, identical byte-for-byte to RQ4_SIGNS[] in
// ggml-quants.c and to every pre-existing per-TU sign literal. Used both as the
// device symbol's initializer (byte-identical default) and as the env-parse
// fallback.
// DEFAULT for this monolith repo: the rand2 pattern (random.seed(2) derived),
// baked in so the no-env-var default already matches the validated rand2 sweep.
#define GGML_CUDA_RQ4_SIGNS_GOLDEN                                    \
    -1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f,                \
    -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f, +1.0f,                \
    +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f,                \
    +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, -1.0f

// TU-local device symbol. `static` => one copy per translation unit. Each TU
// that includes this header gets its own, initialized to the golden literal.
static __constant__ float ggml_cuda_rq4_signs_dev[32] = {
    GGML_CUDA_RQ4_SIGNS_GOLDEN,
};

// Parse RQ4_SIGNS into `out` (32 floats, each ±1). Default (unset/malformed):
// the golden literal. Centralized so every TU agrees on the format and the
// default.
static inline void ggml_cuda_rq4_signs_parse(float out[32]) {
    static const float golden[32] = { GGML_CUDA_RQ4_SIGNS_GOLDEN };
    const char * env = std::getenv("RQ4_SIGNS");
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
#undef GGML_CUDA_RQ4_SIGNS_GOLDEN

// Idempotent host init: parse RQ4_SIGNS once, upload the 32-float pattern to
// THIS TU's `ggml_cuda_rq4_signs_dev` via cudaMemcpyToSymbol. Call from
// any RQ4 host entry point before the first kernel launch. No-op on the
// default path (env unset) beyond the parse + a byte-identical upload.
static inline void ggml_cuda_rq4_init_signs() {
    static bool inited = false;
    if (inited) return;
    float h[32];
    ggml_cuda_rq4_signs_parse(h);
    cudaMemcpyToSymbol(ggml_cuda_rq4_signs_dev, h, sizeof(h));
    inited = true;
}

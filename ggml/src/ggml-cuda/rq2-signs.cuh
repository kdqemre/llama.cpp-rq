// RQ2 sign-pattern parameterization (research sweep).
//
// The RQ2 format applies a per-32-element-subblock randomized Hadamard
// transform whose diagonal sign pattern defaults to the rand2 golden literal
// (byte-identical to RQ2_SIGNS[] in ggml-quants.c). For the sign-pattern sweep
// the pattern is a RUNTIME parameter:
//
//   RQ2_SIGNS = exactly 32 chars of '+'/'-'  (lane-wise +/-1 diagonal)
//
// Default (env unset OR malformed): the golden literal, byte-identical to
// the CPU RQ2_SIGNS[] => bit-identical inference when the env var is unset.
//
// IMPORTANT: the quantizer (llama-quantize) and the inference engine
// (llama-server / llama-cli) are SEPARATE processes. Both must be launched with
// the SAME RQ2_SIGNS so the GGUF's baked rotation matches the runtime activation
// rotation; otherwise the result is garbage.
//
// Mechanism: every CUDA TU that consumes the RQ2 rotation includes this header,
// which defines a TU-local `static __constant__` device symbol initialized to
// the golden literal. Each TU's host entry point calls ggml_cuda_rq2_init_signs()
// once before its first kernel launch.

#pragma once

#include <cstdlib>
#include <cstring>

// Golden-ratio literal, identical byte-for-byte to RQ2_SIGNS[] in ggml-quants.c
// (the RQ2 sign-search WINNER, == the CPU RQ2_SIGNS[] default). Used as the device symbol initializer and
// the env-parse fallback.
#define GGML_CUDA_RQ2_SIGNS_GOLDEN                                    \
    -1.0f, +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f,                \
    -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, +1.0f,                \
    +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f,                \
    +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, -1.0f

// TU-local device symbol. `static` => one copy per translation unit.
static __constant__ float ggml_cuda_rq2_signs_dev[32] = {
    GGML_CUDA_RQ2_SIGNS_GOLDEN,
};

// Parse RQ2_SIGNS into `out` (32 floats, each +/-1). Default: golden literal.
static inline void ggml_cuda_rq2_signs_parse(float out[32]) {
    static const float golden[32] = { GGML_CUDA_RQ2_SIGNS_GOLDEN };
    const char * env = std::getenv("RQ2_SIGNS");
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
#undef GGML_CUDA_RQ2_SIGNS_GOLDEN

// Idempotent host init: parse RQ2_SIGNS once, upload the 32-float pattern to
// THIS TU's ggml_cuda_rq2_signs_dev via cudaMemcpyToSymbol. Call before the
// first kernel launch. No-op beyond the parse + upload on subsequent calls.
static inline void ggml_cuda_rq2_init_signs() {
    static bool inited = false;
    if (inited) return;
    float h[32];
    ggml_cuda_rq2_signs_parse(h);
    cudaMemcpyToSymbol(ggml_cuda_rq2_signs_dev, h, sizeof(h));
    inited = true;
}

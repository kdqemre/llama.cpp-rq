// tests/test-rq2-cpu.cpp
//
// Phase 2 gate for GGML_TYPE_RQ2 (2-bit WHT-rotated K-quant), CPU-only.
//   (a) round-trip NMSE:  quantize_row_rq2_ref -> dequantize_row_rq2 vs original
//                         (2-bit is lossy; expect NMSE well below 0.15 thanks to WHT)
//   (b) vec_dot consistency: ggml_vec_dot_rq2_f32 vs manual dequant+F32 dot
//                         (same dequant math on both paths -> expect ~1e-4)
#include "ggml.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
extern "C" void ggml_vec_dot_rq2_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_cpu_init(void);
static uint32_t g_state = 0xC0FFEE01u;
static float frand() {
    // xorshift32 -> uniform; box-muller-ish normal via sum of uniforms for realistic spread
    uint32_t x = g_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; g_state = x;
    float u = (x >> 8) * (1.0f / 16777216.0f); // [0,1)
    return u;
}

static float nrand() {
    // sum of 3 uniforms ~ triangular; add a heavy tail with prob 0.05
    float v = (frand() + frand() + frand() - 1.5f);
    if (frand() < 0.05f) v *= 5.0f; // inject outliers (WHT rotation target)
    return v;
}
int main(void) {
    g_state = 0xC0FFEE01u;
    ggml_cpu_init(); // populate fp16 lookup table used by GGML_CPU_FP16_TO_FP32
    constexpr int64_t N = 8 * 256; // 8 superblocks of 256
    constexpr int64_t nbytes = N / 256 * sizeof(block_rq2);

    std::vector<float> src(N), deq(N);
    for (int i = 0; i < N; ++i) src[i] = nrand();

    std::vector<uint8_t> q(nbytes);
    quantize_row_rq2_ref(src.data(), (block_rq2 *) q.data(), N);

    // size sanity
    size_t expect_sz = ggml_row_size(GGML_TYPE_RQ2, N);
    if (expect_sz != nbytes) {
        printf("FAIL size: ggml_row_size=%zu nbytes=%zu\n", expect_sz, (size_t)nbytes);
        return 1;
    }
    printf("block_rq2 sizeof=%zu  row_size=%zu  bpw=%.4f\n",
           sizeof(block_rq2), expect_sz, 8.0 * expect_sz / N);

    dequantize_row_rq2((const block_rq2 *) q.data(), deq.data(), N);

    // (a) round-trip NMSE
    double num = 0, den = 0;
    for (int i = 0; i < N; ++i) { double d = src[i] - deq[i]; num += d*d; den += (double)src[i]*src[i]; }
    double nmse_rt = den > 0 ? num / den : num;
    printf("(a) round-trip NMSE = %.6f  (gate < 0.15)\n", nmse_rt);

    // (b) vec_dot consistency: vec_dot_rq2_f32(block, act) vs sum(deq[i]*act[i])
    std::vector<float> act(N);
    g_state = 0x12345678u;
    for (int i = 0; i < N; ++i) act[i] = nrand();

    float s_test = 0.0f;
    ggml_vec_dot_rq2_f32((int) N, &s_test, 0, q.data(), 0, act.data(), 0, 1);

    double ref = 0.0;
    for (int i = 0; i < N; ++i) ref += (double) deq[i] * act[i];
    double rel = fabs(ref - s_test) / (fabs(ref) + 1e-30);
    printf("(b) vec_dot: test=%.6f ref=%.6f relerr=%.3e  (gate < 1e-4)\n", s_test, ref, rel);

    bool ok = (nmse_rt < 0.15) && (rel < 1e-4) && (expect_sz == (size_t)nbytes);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

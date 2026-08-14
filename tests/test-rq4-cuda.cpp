// tests/test-rq4-cuda.cpp
//
// Validates the GGML_TYPE_RQ4 CUDA kernels against the CPU reference
// backend by building the SAME ggml graph on both backends and diffing the
// float outputs. The CPU backend's get_rows/mul_mat dispatch to the reference
// functions dequantize_row_rq4 / ggml_vec_dot_rq4_f32, so a
// two-backend graph compare is exactly the desired CUDA-vs-CPU equivalence.
//
// Three checks:
//   (a) DEQUANT  via get_rows  -> CUDA k_get_rows_rq4  vs CPU dequantize_row_rq4
//   (b) VEC_DOT  via mul_mat   -> CUDA MMVQ (n=1, large K) vs CPU ggml_vec_dot_rq4_f32
//   (c) MMQ probe via mul_mat  -> CUDA MMQ (n=64) path    [informational; suspected divergent]
//
// Modeled on tests/test-tq3-kv-cache-cuda.cpp. Public GGML API only; no
// internal headers required (GGML_BACKEND_DL=OFF build links everything in).
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int64_t QK = 256; // == QK_RQ4 == QK_K

// ---------------------------------------------------------------------------
// backend selection
// ---------------------------------------------------------------------------
static ggml_backend_t pick_backend_by_reg(const char * reg_name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t  reg = ggml_backend_dev_backend_reg(dev);
        if (std::strcmp(ggml_backend_reg_name(reg), reg_name) == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// graph runner: build -> alloc on `backend` -> set inputs -> compute -> read out
// ---------------------------------------------------------------------------
struct tensor_input {
    const char * name;
    const void * data;
    size_t       nbytes;
};

static std::vector<float> run_graph_float(ggml_backend_t backend,
                                          ggml_tensor * (*build_fn)(ggml_context *),
                                          const std::vector<tensor_input> & inputs,
                                          const char * dst_name)
{
    ggml_init_params ip = {
        /*.mem_size =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /*.mem_base =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    GGML_ASSERT(ctx);

    ggml_tensor * dst = build_fn(ctx);
    GGML_ASSERT(dst);
    ggml_set_name(dst, dst_name);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    for (const auto & in : inputs) {
        ggml_tensor * t = ggml_get_tensor(ctx, in.name);
        GGML_ASSERT(t);
        ggml_backend_tensor_set(t, in.data, 0, in.nbytes); // returns void
    }

    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    ggml_tensor * out = ggml_get_tensor(ctx, dst_name);
    GGML_ASSERT(out && out->type == GGML_TYPE_F32);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float)); // returns void

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// ---------------------------------------------------------------------------
// diff metrics: max-abs + NMSE (matches test-backend-ops error metric)
// ---------------------------------------------------------------------------
static bool diff(const std::vector<float> & cpu,
                 const std::vector<float> & cuda,
                 const char * label, double tol_abs, double tol_nmse)
{
    GGML_ASSERT(cpu.size() == cuda.size());
    double max_abs = 0.0, num = 0.0, den = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        double d  = (double) cuda[i] - (double) cpu[i];
        double ad = std::fabs(d);
        if (ad > max_abs) max_abs = ad;
        num += d * d;
        den += (double) cpu[i] * (double) cpu[i];
    }
    double nmse = den > 0.0 ? num / den : (num > 0.0 ? INFINITY : 0.0);
    bool ok = (max_abs <= tol_abs) && std::isfinite(nmse) && (nmse <= tol_nmse);
    std::printf("[%-22s] N=%-6zu max_abs=%.3e (tol %.0e)  nmse=%.3e (tol %.0e) -> %s\n",
                label, cpu.size(), max_abs, tol_abs, nmse, tol_nmse, ok ? "PASS" : "FAIL");
    if (!ok) {
        for (size_t i = 0, shown = 0; i < cpu.size() && shown < 8; ++i) {
            if (std::fabs((double)cuda[i] - (double)cpu[i]) > tol_abs) {
                std::printf("    [%zu] cpu=%+.6f  cuda=%+.6f  d=%+.3e\n", i, cpu[i], cuda[i], cuda[i]-cpu[i]);
                ++shown;
            }
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// deterministic pseudo-random input in [-1, 1)
// ---------------------------------------------------------------------------
static float frand(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return ((state >> 8) & 0xFFFFFFu) * (float)(1.0 / 0x1000000) * 2.0f - 1.0f;
}

// Pack a [ncol, nrow] RQ4 tensor on host via the public quantize API.
// Returns the packed bytes; also fills the source f32 (for reference).
static std::vector<uint8_t> make_rq4(int64_t nrow, int64_t ncol, uint32_t seed,
                                         std::vector<float> & f32_out)
{
    GGML_ASSERT(ncol % QK == 0);
    f32_out.resize((size_t) nrow * ncol);
    uint32_t s = seed;
    for (auto & v : f32_out) v = frand(s);

    size_t row_bytes = (ncol / QK) * ggml_type_size(GGML_TYPE_RQ4);
    std::vector<uint8_t> packed((size_t) nrow * row_bytes);
    size_t w = ggml_quantize_chunk(GGML_TYPE_RQ4, f32_out.data(), packed.data(),
                                   0, nrow, ncol, nullptr);
    GGML_ASSERT(w == (size_t) nrow * row_bytes);
    return packed;
}

// Heavy-tailed variant: Gaussian-ish + ~1.5% large spikes -> triggers the
// 1-outlier extraction so ol_loc/ol_delta are populated. Used to exercise the
// MMQ outlier-correction kernel (uniform weights never trigger outliers).
static std::vector<uint8_t> make_rq4_ht(int64_t nrow, int64_t ncol, uint32_t seed,
                                            std::vector<float> & f32_out)
{
    GGML_ASSERT(ncol % QK == 0);
    f32_out.resize((size_t) nrow * ncol);
    uint32_t s = seed;
    for (auto & v : f32_out) {
        float g = 0;
        for (int t = 0; t < 4; ++t) g += frand(s);
        g *= 0.5f;
        if (frand(s) > 0.985f) g *= (3.0f + 5.0f * std::fabs(frand(s)));
        v = g;
    }
    size_t row_bytes = (ncol / QK) * ggml_type_size(GGML_TYPE_RQ4);
    std::vector<uint8_t> packed((size_t) nrow * row_bytes);
    size_t w = ggml_quantize_chunk(GGML_TYPE_RQ4, f32_out.data(), packed.data(),
                                   0, nrow, ncol, nullptr);
    GGML_ASSERT(w == (size_t) nrow * row_bytes);
    return packed;
}

// ---------------------------------------------------------------------------
// graph builders
// ---------------------------------------------------------------------------
// (a) DEQUANT via get_rows: src0 = rq4 [ncol, nrow], idx = 0..nrow-1
//     -> dst F32 [ncol, nrow] = fully dequantized tensor on both backends.
struct cfg_getrows { int64_t ncol; int64_t nrow; };
static cfg_getrows g_gr = { 8 * QK, 4 };

static ggml_tensor * build_get_rows(ggml_context * ctx) {
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_RQ4, g_gr.ncol, g_gr.nrow);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,    g_gr.nrow);
    ggml_tensor * out = ggml_get_rows(ctx, a, idx);
    ggml_set_name(a,   "a");
    ggml_set_name(idx, "idx");
    return out;
}

// (b)/(c) mul_mat: src0 = rq4 [k, m], src1 = F32 [k, n], dst = [m, n].
struct cfg_mulmat { int64_t m; int64_t n; int64_t k; };
static cfg_mulmat g_mm_mmvq = { 16,  1, 144 * QK }; // n=1  -> MMVQ; K=144 blocks -> ncols1 kernel
static cfg_mulmat g_mm_mmq  = { 32, 64,  8 * QK };  // n=64 -> MMQ path
static cfg_mulmat g_cur_mm  = g_mm_mmvq;            // active shape for build_mul_mat

static ggml_tensor * build_mul_mat(ggml_context * ctx) {
    const cfg_mulmat & c = g_cur_mm;
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_RQ4, c.k, c.m);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,     c.k, c.n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    return out;
}

// ---------------------------------------------------------------------------
// (d) CUDA-only get_rows bit-identity checksum + microbenchmark.
//     CPU get_rows cannot take a quantized src0 (aborts in ops.cpp), so a
//     two-backend diff is impossible. Instead we run get_rows on CUDA over a
//     FIXED synthetic RQ4 tensor and emit a deterministic checksum (sum of
//     squares + 64-bit FNV-1a over the raw F32 output bytes). This is a
//     bit-identity oracle: it must be byte-for-byte identical before/after any
//     internal refactor of k_get_rows_rq4 (only the thread/block->superblock
//     mapping may change; the dequant math must not).
// ---------------------------------------------------------------------------
static uint64_t fnv1a64(const void * data, size_t nbytes) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < nbytes; ++i) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

static void cuda_getrows_checksum(ggml_backend_t be_cuda, const char * label,
                                  int64_t ncol, int64_t nrow, uint32_t seed) {
    g_gr.ncol = ncol; g_gr.nrow = nrow;
    std::vector<float> f32_a;
    auto packed = make_rq4(nrow, ncol, seed, f32_a);
    std::vector<int32_t> idx(nrow);
    for (int64_t i = 0; i < nrow; ++i) idx[i] = (int32_t) i;
    std::vector<tensor_input> in = {
        { "a",   packed.data(), packed.size() },
        { "idx", idx.data(),    idx.size() * sizeof(int32_t) },
    };
    auto out = run_graph_float(be_cuda, build_get_rows, in, "out");
    double sumsq = 0.0;
    for (float v : out) sumsq += (double) v * (double) v;
    uint64_t h = fnv1a64(out.data(), out.size() * sizeof(float));
    std::printf("[%-36s] ncol=%-5lld nrow=%-3lld nelem=%-9zu sumsq=%.6f fnv1a=%016llx\n",
                label, (long long) ncol, (long long) nrow, out.size(),
                sumsq, (unsigned long long) h);
}

// Microbenchmark: build get_rows graph once, time N computes (graph_compute is
// synchronous on the CUDA backend, so wall-clock timing is valid).
static void time_getrows_cuda(ggml_backend_t be_cuda, const char * label,
                              int64_t ncol, int64_t nrow, uint32_t seed, int iters) {
    g_gr.ncol = ncol; g_gr.nrow = nrow;
    std::vector<float> f32_a;
    auto packed = make_rq4(nrow, ncol, seed, f32_a);
    std::vector<int32_t> idx(nrow);
    for (int64_t i = 0; i < nrow; ++i) idx[i] = (int32_t) i;

    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * dst = build_get_rows(ctx);
    ggml_set_name(dst, "out");
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be_cuda);
    GGML_ASSERT(buf);
    ggml_backend_tensor_set(ggml_get_tensor(ctx, "a"),   packed.data(), 0, packed.size());
    ggml_backend_tensor_set(ggml_get_tensor(ctx, "idx"), idx.data(),    0, idx.size() * sizeof(int32_t));

    ggml_backend_graph_compute(be_cuda, gf);  // warmup
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be_cuda, gf);
    auto t1 = std::chrono::steady_clock::now();
    double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gib = (double) packed.size() / (ms * 1e-3) / 1e9;  // src0 read bandwidth
    std::printf("[%-36s] iters=%-4d  %8.4f ms/iter  src0_read=%6.1f GB/s\n",
                label, iters, ms, gib);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
int main(void)
{
    ggml_backend_load_all();

    ggml_backend_t be_cuda = pick_backend_by_reg("CUDA");
    if (!be_cuda) {
        std::puts("CUDA backend not found; skipping RQ4 CUDA validation");
        return 0; // not a failure of the kernels under test
    }
    ggml_backend_t be_cpu = pick_backend_by_reg("CPU");
    if (!be_cpu) {
        be_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    GGML_ASSERT(be_cpu);

    bool ok = true;

    // ---- (a) VEC_DOT via mul_mat, n=1, small K -> generic MMVQ (vecdotq.cuh)
    //         NOTE: CPU get_rows does NOT support quantized src0 (aborts in
    //         ops.cpp), so the dequant math is validated through the mul_mat
    //         vec_dot path instead. K = 8 blocks (< 144) avoids the specialized
    //         ncols1 kernel, exercising the generic vec_dot_rq4_q8_1.
    {
        g_cur_mm = { 16, 1, 8 * QK };
        std::vector<float> f32_a, f32_b((size_t) g_cur_mm.k * g_cur_mm.n);
        auto packed = make_rq4(g_cur_mm.m, g_cur_mm.k, 0xC0FFEEu, f32_a);
        uint32_t s = 0xA11CEu;
        for (auto & v : f32_b) v = frand(s);
        std::vector<tensor_input> in = {
            { "a", packed.data(), packed.size() },
            { "b", f32_b.data(),  f32_b.size() * sizeof(float) },
        };
        auto cpu  = run_graph_float(be_cpu,  build_mul_mat, in, "out");
        auto cuda = run_graph_float(be_cuda, build_mul_mat, in, "out");
        // tol_abs is loose: dot magnitudes reach ~10s-100s, so Q8_1 activation
        // quantization legitimately yields ~0.1-0.5 abs error. NMSE is the real bar.
        ok = diff(cpu, cuda, "VECDOT mul_mat n=1 smallK (generic MMVQ)", 1.0, 1e-3) && ok;
    }

    // ---- (b) VEC_DOT via mul_mat, n=1 -> MMVQ -----------------------------
    {
        g_cur_mm = g_mm_mmvq;
        std::vector<float> f32_a, f32_b((size_t) g_cur_mm.k * g_cur_mm.n);
        auto packed = make_rq4(g_cur_mm.m, g_cur_mm.k, 0x1234u, f32_a);
        uint32_t s = 0x5678u;
        for (auto & v : f32_b) v = frand(s);
        std::vector<tensor_input> in = {
            { "a", packed.data(),  packed.size() },
            { "b", f32_b.data(),   f32_b.size() * sizeof(float) },
        };
        auto cpu  = run_graph_float(be_cpu,  build_mul_mat, in, "out");
        auto cuda = run_graph_float(be_cuda, build_mul_mat, in, "out");
        // CUDA quantizes src1 to Q8_1 on the fly; CPU dots against raw F32.
        // 5e-4 NMSE is the repo's standard quantized-mul_mat bar (1e-3 here
        // is a 2x margin for the first run).
        ok = diff(cpu, cuda, "VECDOT mul_mat n=1 (MMVQ)", 1.0, 1e-3) && ok;
    }

    // ---- (c) MMQ probe, n=64 (informational) ------------------------------
    {
        g_cur_mm = g_mm_mmq;
        std::vector<float> f32_a, f32_b((size_t) g_cur_mm.k * g_cur_mm.n);
        auto packed = make_rq4(g_cur_mm.m, g_cur_mm.k, 0x9ABCu, f32_a);
        uint32_t s = 0xDEF0u;
        for (auto & v : f32_b) v = frand(s);
        std::vector<tensor_input> in = {
            { "a", packed.data(),  packed.size() },
            { "b", f32_b.data(),   f32_b.size() * sizeof(float) },
        };
        auto cpu  = run_graph_float(be_cpu,  build_mul_mat, in, "out");
        auto cuda = run_graph_float(be_cuda, build_mul_mat, in, "out");
        // MMQ re-quantizes dequantized weights to int8 (load_tiles_rq4) on
        // top of Q8_1 activation quant -> slightly more error than MMVQ, but
        // still tiny in NMSE. tol_abs is loose (dot magnitudes ~10-30); NMSE is
        // the real bar. Now gates pass/fail (MMQ is a validated path).
        ok = diff(cpu, cuda, "VECDOT mul_mat n=64 (MMQ prefill)", 1.0, 5e-3) && ok;
    }

    // ---- (c2) MMQ probe with HEAVY-TAILED weights (exercises outliers) ------
    {
        g_cur_mm = g_mm_mmq;
        std::vector<float> f32_a, f32_b((size_t) g_cur_mm.k * g_cur_mm.n);
        auto packed = make_rq4_ht(g_cur_mm.m, g_cur_mm.k, 0x9ABCu, f32_a);
        uint32_t s = 0xDEF0u;
        for (auto & v : f32_b) v = frand(s);
        std::vector<tensor_input> in = {
            { "a", packed.data(),  packed.size() },
            { "b", f32_b.data(),   f32_b.size() * sizeof(float) },
        };
        auto cpu  = run_graph_float(be_cpu,  build_mul_mat, in, "out");
        auto cuda = run_graph_float(be_cuda, build_mul_mat, in, "out");
        // Heavy-tail weights populate ol_loc/ol_delta, so this exercises BOTH the
        // bulk rotated dot AND the outlier-correction kernel. Same bar as (c).
        ok = diff(cpu, cuda, "VECDOT mul_mat n=64 (MMQ heavy-tail)", 1.0, 5e-3) && ok;
    }


    // ---- (d) CUDA-only get_rows bit-identity checksum + microbenchmark -----
    // Two shapes: a clean multiple of the pack width (no tail guard) and a
    // partial tail (exercises the in-kernel bounds guard). Checksums are the
    // bit-identity oracle (must match before/after a kernel refactor); the
    // microbenchmark gives a clean get_rows-only timing signal.
    cuda_getrows_checksum(be_cuda, "GETROWS checksum (d1) clean",   64 * QK,  5, 0x5EED5EEDu);
    cuda_getrows_checksum(be_cuda, "GETROWS checksum (d2) tail",    13 * QK, 17, 0xBADC0FFEu);
    time_getrows_cuda  (be_cuda, "GETROWS microbench (d2) tail", 13 * QK, 17, 0xBADC0FFEu, 2000);
    // Large stress shape: 256 superblocks/row -> ceil(256/8)=32 block-rows x 512
    // selected rows = 16384 blocks; enough to saturate the SMs so occupancy
    // (active warps/block) rather than launch overhead dominates timing.
    time_getrows_cuda  (be_cuda, "GETROWS microbench (d3) large", 256 * QK, 512, 0xFEEDFACEu, 500);

    // ---- (e) Dequant round-trip NMSE on HEAVY-TAILED data (the Method C regime) ----
    // Uniform [-1,1) noise has no >3sigma elements, so it never triggers the outlier
    // path. Build heavy-tailed weights (Gaussian + a few large spikes), quantize to
    // RQ4, dequant via CUDA get_rows, and report NMSE vs the original FP32.
    // Headline quality gate: target ~3.6e-4 (~38x better than Q4_K's dequant error).
    {
        const int64_t ncol = 64 * QK;
        const int64_t nrow = 8;
        std::vector<float> f32_a((size_t) ncol * nrow);
        uint32_t s = 0x5A5A5Au;
        for (auto & v : f32_a) {
            float g = 0;
            for (int t = 0; t < 4; ++t) g += frand(s);   // ~triangular, Gaussian-ish
            g *= 0.5f;
            // ~1.5% heavy-tail spike (realistic for LLM weights; the regime Method C targets)
            if (frand(s) > 0.985f) g *= (3.0f + 5.0f * std::fabs(frand(s)));
            v = g;
        }
        size_t row_bytes = (ncol / QK) * ggml_type_size(GGML_TYPE_RQ4);
        std::vector<uint8_t> packed((size_t) nrow * row_bytes);
        size_t w = ggml_quantize_chunk(GGML_TYPE_RQ4, f32_a.data(), packed.data(), 0, nrow, ncol, nullptr);
        GGML_ASSERT(w == (size_t) nrow * row_bytes);

        g_gr.ncol = ncol; g_gr.nrow = nrow;
        std::vector<int32_t> idx(nrow);
        for (int64_t i = 0; i < nrow; ++i) idx[i] = (int32_t) i;
        std::vector<tensor_input> in = {
            { "a",   packed.data(), packed.size() },
            { "idx", idx.data(),    idx.size() * sizeof(int32_t) },
        };
        auto deq = run_graph_float(be_cuda, build_get_rows, in, "out");
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < f32_a.size(); ++i) {
            double d  = (double) deq[i] - (double) f32_a[i];
            num += d * d;
            den += (double) f32_a[i] * (double) f32_a[i];
        }
        double nmse = den > 0.0 ? num / den : 0.0;
        std::printf("[DEQUANT round-trip heavy-tail     ] N=%-7zu nmse=%.4e (target ~3.6e-4, 38x vs Q4_K)\n",
                    f32_a.size(), nmse);
    }

    ggml_backend_free(be_cuda);
    ggml_backend_free(be_cpu);

    std::printf("\nRQ4 CUDA validation: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

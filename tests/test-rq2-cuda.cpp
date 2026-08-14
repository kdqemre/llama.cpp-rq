// tests/test-rq2-cuda.cpp
//
// Validates the GGML_TYPE_RQ2 CUDA kernels against the CPU reference backend by
// building the SAME ggml graph on both backends and diffing the float outputs.
//
// Four NMSE checks vs CPU:
//   (a) VEC_DOT generic MMVQ:  n=1, small K -> vec_dot_rq2_q8_1_rot
//   (b) ncols1:                n=1, large K -> generic ncols1 path
//   (c) PREFILL (MMQ):         n=64         -> MMQ dp4a (THE gibberish gate)
//   (d) DEQUANT via get_rows   -> k_get_rows_rq2 vs CPU + round-trip NMSE
//
// Modeled on tests/test-rq3-k-l-cuda.cpp. Public GGML API only.
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int64_t QK = 256; // == QK_RQ2 == QK_K

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

struct tensor_input { const char * name; const void * data; size_t nbytes; };

static std::vector<float> run_graph_float(ggml_backend_t backend,
                                          ggml_tensor * (*build_fn)(ggml_context *),
                                          const std::vector<tensor_input> & inputs,
                                          const char * dst_name)
{
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
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
        ggml_backend_tensor_set(t, in.data, 0, in.nbytes);
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_tensor * out = ggml_get_tensor(ctx, dst_name);
    GGML_ASSERT(out && out->type == GGML_TYPE_F32);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

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
    std::printf("[%-38s] N=%-6zu max_abs=%.3e (tol %.0e)  nmse=%.3e (tol %.0e) -> %s\n",
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

static float frand(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return ((state >> 8) & 0xFFFFFFu) * (float)(1.0 / 0x1000000) * 2.0f - 1.0f;
}

// Pack a [ncol, nrow] RQ2 tensor on host via the public quantize API.
static std::vector<uint8_t> make_rq2(int64_t nrow, int64_t ncol, uint32_t seed,
                                         std::vector<float> & f32_out)
{
    GGML_ASSERT(ncol % QK == 0);
    f32_out.resize((size_t) nrow * ncol);
    uint32_t s = seed;
    for (auto & v : f32_out) v = frand(s);
    size_t row_bytes = (ncol / QK) * ggml_type_size(GGML_TYPE_RQ2);
    std::vector<uint8_t> packed((size_t) nrow * row_bytes);
    size_t w = ggml_quantize_chunk(GGML_TYPE_RQ2, f32_out.data(), packed.data(),
                                   0, nrow, ncol, nullptr);
    GGML_ASSERT(w == (size_t) nrow * row_bytes);
    return packed;
}

// mul_mat: src0 = rq2 [k, m], src1 = F32 [k, n], dst = [m, n].
struct cfg_mulmat { int64_t m; int64_t n; int64_t k; };
static cfg_mulmat g_cur_mm = { 16, 1, 8 * QK };

static ggml_tensor * build_mul_mat(ggml_context * ctx) {
    const cfg_mulmat & c = g_cur_mm;
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_RQ2, c.k, c.m);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, c.n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    return out;
}

// DEQUANT via get_rows: src0 = rq2 [ncol, nrow], idx = 0..nrow-1
struct cfg_getrows { int64_t ncol; int64_t nrow; };
static cfg_getrows g_gr = { 8 * QK, 4 };

static ggml_tensor * build_get_rows(ggml_context * ctx) {
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_RQ2, g_gr.ncol, g_gr.nrow);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,    g_gr.nrow);
    ggml_tensor * out = ggml_get_rows(ctx, a, idx);
    ggml_set_name(a,   "a");
    ggml_set_name(idx, "idx");
    return out;
}

int main(void)
{
    ggml_backend_load_all();
    ggml_backend_t be_cuda = pick_backend_by_reg("CUDA");
    if (!be_cuda) { std::puts("CUDA backend not found; skipping RQ2 CUDA validation"); return 0; }
    ggml_backend_t be_cpu = pick_backend_by_reg("CPU");
    if (!be_cpu) be_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(be_cpu);

    bool ok = true;

    // (a) generic MMVQ: n=1, small K (8 blocks) -> vec_dot_rq2_q8_1_rot
    {
        g_cur_mm = { 16, 1, 8 * QK };
        std::vector<float> f32_a, f32_b((size_t) g_cur_mm.k * g_cur_mm.n);
        auto packed = make_rq2(g_cur_mm.m, g_cur_mm.k, 0xC0FFEEu, f32_a);
        uint32_t s = 0xA11CEu;
        for (auto & v : f32_b) v = frand(s);
        std::vector<tensor_input> in = {
            { "a", packed.data(), packed.size() },
            { "b", f32_b.data(),  f32_b.size() * sizeof(float) },
        };
        auto cpu  = run_graph_float(be_cpu,  build_mul_mat, in, "out");
        auto cuda = run_graph_float(be_cuda, build_mul_mat, in, "out");
        ok = diff(cpu, cuda, "VECDOT mul_mat n=1 smallK (generic MMVQ)", 1.0, 1e-3) && ok;
    }

    // (b) ncols1: n=1, large K -> generic ncols1 path
    {
        g_cur_mm = { 16, 1, 64 * QK };
        std::vector<float> f32_a, f32_b((size_t) g_cur_mm.k * g_cur_mm.n);
        auto packed = make_rq2(g_cur_mm.m, g_cur_mm.k, 0x1234u, f32_a);
        uint32_t s = 0x5678u;
        for (auto & v : f32_b) v = frand(s);
        std::vector<tensor_input> in = {
            { "a", packed.data(), packed.size() },
            { "b", f32_b.data(),  f32_b.size() * sizeof(float) },
        };
        auto cpu  = run_graph_float(be_cpu,  build_mul_mat, in, "out");
        auto cuda = run_graph_float(be_cuda, build_mul_mat, in, "out");
        ok = diff(cpu, cuda, "VECDOT mul_mat n=1 largeK (ncols1)", 1.0, 1e-3) && ok;
    }

    // (c) prefill MMQ: n=64 -> MMQ dp4a path (THE gibberish gate)
    {
        g_cur_mm = { 32, 64, 8 * QK };
        std::vector<float> f32_a, f32_b((size_t) g_cur_mm.k * g_cur_mm.n);
        auto packed = make_rq2(g_cur_mm.m, g_cur_mm.k, 0x9ABCu, f32_a);
        uint32_t s = 0xDEF0u;
        for (auto & v : f32_b) v = frand(s);
        std::vector<tensor_input> in = {
            { "a", packed.data(), packed.size() },
            { "b", f32_b.data(),  f32_b.size() * sizeof(float) },
        };
        auto cpu  = run_graph_float(be_cpu,  build_mul_mat, in, "out");
        auto cuda = run_graph_float(be_cuda, build_mul_mat, in, "out");
        ok = diff(cpu, cuda, "VECDOT mul_mat n=64 (prefill MMQ dp4a)", 1.0, 1e-3) && ok;
    }

    // (d) DEQUANT via get_rows: CUDA k_get_rows_rq2 vs CPU dequantize_row_rq2
    {
        g_gr = { 64 * QK, 8 };
        std::vector<float> f32_a;
        auto packed = make_rq2(g_gr.nrow, g_gr.ncol, 0x5A5Au, f32_a);
        std::vector<int32_t> idx(g_gr.nrow);
        for (int64_t i = 0; i < g_gr.nrow; ++i) idx[i] = (int32_t) i;
        std::vector<tensor_input> in = {
            { "a",   packed.data(), packed.size() },
            { "idx", idx.data(),    idx.size() * sizeof(int32_t) },
        };
        auto cpu  = run_graph_float(be_cpu,  build_get_rows, in, "out");
        auto cuda = run_graph_float(be_cuda, build_get_rows, in, "out");
        ok = diff(cpu, cuda, "DEQUANT get_rows vs CPU (exact)", 2e-3, 1e-6) && ok;

        // round-trip NMSE vs original FP32
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < f32_a.size(); ++i) {
            double d = (double) cuda[i] - (double) f32_a[i];
            num += d * d; den += (double) f32_a[i] * (double) f32_a[i];
        }
        double nmse = den > 0.0 ? num / den : 0.0;
        std::printf("[DEQUANT round-trip NMSE              ] N=%-7zu nmse=%.4e\n", f32_a.size(), nmse);
    }

    ggml_backend_free(be_cuda);
    ggml_backend_free(be_cpu);
    std::printf("\nRQ2 CUDA validation: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

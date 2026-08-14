// RQ3 fused activation prep (Kernel A) + batched activation rotation.
// Format-agnostic: identical math to rq4-prep.cu, reading the independent
// ggml_cuda_rq3_signs_dev sign source.

#include "common.cuh"
#include "rq3-prep.cuh"
#include "rq3-signs.cuh"

// ---------------------------------------------------------------------------
// RQ3 MMQ activation rotation. Fused copy + forward randomized Hadamard
// transform per 32-element sub-block (src -> dst), so the standard q8_1
// quantizer then produces ROTATED q8_1. Identity: <RHT_inverse(w_rot), a> =
// <w_rot, RHT_forward(a)> -> the WHT moves off the weights onto the activation,
// amortized once/token across all weight rows.
// ---------------------------------------------------------------------------
static __global__ void rq3_copy_rotate_act_kernel(
        float * __restrict__ dst, const float * __restrict__ src, int64_t n) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int64_t base = ((int64_t)blockIdx.x * (blockDim.x >> 5) + warp_id) * 32;
    if (base >= n) return;
    float val = src[base + lane] * ggml_cuda_rq3_signs_dev[lane];
#pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step, 32);
        val = (lane & step) ? (other - val) : (other + val);
    }
    static constexpr float inv_sqrt32 = 0.1767766952966369f; // 1/sqrt(32)
    dst[base + lane] = val * inv_sqrt32;
}

void ggml_cuda_rq3_rotate_act(
        float * dst, const float * src, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    ggml_cuda_rq3_init_signs();
    constexpr int n_warps = 8;  // 256 threads/block
    const int64_t n_blocks = (n / 32 + n_warps - 1) / n_warps;
    rq3_copy_rotate_act_kernel<<<n_blocks, 32*n_warps, 0, stream>>>(dst, src, n);
}

// ---------------------------------------------------------------------------
// RQ3 fused activation prep (Kernel A): replaces memcpy + rotate + quantize
// + fixup with a single launch. Grid/indexing mirrors quantize_q8_1 so the
// output q8_1 buffer is layout-identical, but per 32-element sub-block (== one
// q8_1 block == one warp) it:
//   1. warp-reduces Sa = sum of the ORIGINAL activation (carried for the min term),
//   2. applies the forward randomized Hadamard transform in-register (signs +
//      32-point butterfly + 1/sqrt32) via warp shuffles,
//   3. q8_1-quantizes the ROTATED values (d = amax/127),
//   4. writes ds = (d_rotated, 0) -- Sa lives in the FP32 sidecar `sa_out`.
// Math: <RHT_inverse(c), a> = <c, RHT_forward(a)>.
// ---------------------------------------------------------------------------
static __global__ void rq3_prep_act_kernel(
        const float * __restrict__ x_ptr, block_q8_1 * __restrict__ vy, float * __restrict__ sa_out,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {

    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT x = x_ptr;
    block_q8_1  * GGML_CUDA_RESTRICT y = vy;
    const int64_t i0 = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3 * ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t i00 = i0;
    const int64_t i01 = i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;
    const int64_t i_cont = ((i3 * ne2.z + i2) * ne1 + i1) * ne0 + i0;
    const int64_t ib  = i_cont / QK8_1;     // q8_1 block == one 32-element sub-block
    const int64_t iqs = i_cont % QK8_1;     // == threadIdx.x & 31 (lane within the sub-block)
    const int lane = (int) iqs;

    ggml_cuda_pdl_sync();
    // original activation value; 0 in the padded tail beyond ne00 (matches quantize_q8_1).
    const float av = i0 < ne00 ? x[i03 * s03 + i02 * s02 + i01 * s01 + i00] : 0.0f;

    // Sa = sum of the ORIGINAL activation over this 32-element sub-block.
    float Sa = warp_reduce_sum<QK8_1>(av);
    if (lane == 0) {
        sa_out[ib] = Sa;   // FP32 sidecar: full-precision min-term sum
    }

    // Forward randomized Hadamard transform (32-point): signs + butterfly + 1/sqrt32.
    float val = av * ggml_cuda_rq3_signs_dev[lane];
#pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step, 32);
        val = (lane & step) ? (other - val) : (other + val);
    }
    static constexpr float inv_sqrt32 = 0.1767766952966369f; // 1/sqrt(32)
    val *= inv_sqrt32;

    // q8_1 quantize over the 32 rotated lanes (d = amax/127, q = round(val/d)).
    float amax = warp_reduce_max<QK8_1>(fabsf(val));
    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? (int8_t) 0 : (int8_t) roundf(val / d);

    y[ib].qs[lane] = q;
    if (lane == 0) {
        y[ib].ds = make_half2(d, 0.0f);   // s field unused; Sa lives in the FP32 sidecar
    }
}

void ggml_cuda_rq3_prep_act(
        const float * x, block_q8_1 * vy, float * sa_out,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne0 % QK8_1 == 0);
    ggml_cuda_rq3_init_signs();

    constexpr int prep_block_size = 256; // mult of 32 + divides MATRIX_ROW_PADDING
    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);
    const int64_t block_num_x = (ne0 + prep_block_size - 1) / prep_block_size;
    const dim3 num_blocks(block_num_x, ne1, ne2 * ne3);
    const dim3 block_size(prep_block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(num_blocks, block_size, 0, stream);
    ggml_cuda_kernel_launch(rq3_prep_act_kernel, launch_params,
                            x, vy, sa_out, ne00, s01, s02, s03, ne0, (uint32_t) ne1, ne2_fastdiv);
}

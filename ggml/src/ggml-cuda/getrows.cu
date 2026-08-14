#include "getrows.cuh"
#include "dequantize.cuh"
#include "convert.cuh"
#include "rq4-signs.cuh"
#include "rq3-signs.cuh"
#include "rq2-signs.cuh"

__constant__ static const float rq4_centroids_getrows_cuda[16] = {
    -2.732127f, -2.068495f, -1.617513f, -1.255729f,
    -0.941906f, -0.656424f, -0.387837f, -0.128323f,
     0.128323f,  0.387837f,  0.656424f,  0.941906f,
     1.255729f,  1.617513f,  2.068495f,  2.732127f
};
// RQ4 sign pattern now comes from ggml_cuda_rq4_signs_dev (rq4-signs.cuh),
// parameterized by the RQ4_SIGNS env var with a byte-identical golden default.

// RQ4 get_rows CUDA kernel: 256-element superblock, 32 lanes/warp, WHT per sub-block.
// OCCUPANCY: pack SUPERBLOCKS_PER_BLOCK (=8) superblocks per block, one warp each.
template <typename dst_t>
static __global__ void k_get_rows_rq4(
        const block_rq4 * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    constexpr int SUPERBLOCKS_PER_BLOCK = 8;  // == blockDim.y; one warp per superblock
    const int lane = threadIdx.x;             // 0..31, one element per lane
    const int spw  = threadIdx.y;             // 0..7, which superblock within this block
    const int64_t z = blockIdx.z;

    if (z >= ne11*ne12) { return; }
    const int64_t nblocks = ne00 / QK_RQ4;
    const int64_t b = (int64_t) blockIdx.y * SUPERBLOCKS_PER_BLOCK + spw;  // recomputed superblock index
    if (b >= nblocks) { return; }

    const int i10 = blockIdx.x;
    const int i11 = z / ne12;
    const int i12 = z % ne12;
    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
    const char * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
    const block_rq4 * x = (const block_rq4 *) src0_row + b;
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const float d   = __low2float(x->dm);
    const float dmin = __high2float(x->dm);

    // Per-warp scratch so the SUPERBLOCKS_PER_BLOCK warps never clobber each other.
    __shared__ float shbuf[SUPERBLOCKS_PER_BLOCK * 32];
    float * sb = shbuf + spw * 32;

    for (int sp = 0; sp < 4; sp++) {
        const int il = sp;
        const int is = 2 * sp;
        const uint8_t * q = x->qs + 32 * il;

        uint8_t sc, m;
        get_scale_min_k4(is + 0, x->scales, sc, m);
        const float d1 = d * sc; const float m1 = dmin * m;
        get_scale_min_k4(is + 1, x->scales, sc, m);
        const float d2 = d * sc; const float m2 = dmin * m;

        // Low nibble → sub-block is
        // Uniform dequant in rotated domain (d1*L - m1, min INSIDE inverse WHT).
        sb[lane] = d1 * (q[lane] & 0xF) - m1;
        __syncthreads();
        float val = sb[lane];
        for (int step = 1; step < 32; step <<= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, val, step, 32);
            val = (lane & step) ? (other - val) : (other + val);
        }
        val = val * (ggml_cuda_rq4_signs_dev[lane] / sqrtf(32.0f));
        dst_row[b * QK_RQ4 + 64*il + lane] = ggml_cuda_cast<dst_t>(val);

        // High nibble → sub-block is+1
        __syncthreads();
        sb[lane] = d2 * (q[lane] >> 4) - m2;
        __syncthreads();
        val = sb[lane];
        for (int step = 1; step < 32; step <<= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, val, step, 32);
            val = (lane & step) ? (other - val) : (other + val);
        }
        val = val * (ggml_cuda_rq4_signs_dev[lane] / sqrtf(32.0f));
        dst_row[b * QK_RQ4 + 64*il + 32 + lane] = ggml_cuda_cast<dst_t>(val);
    }
}

template <typename dst_t>
static void get_rows_cuda_rq4(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_RQ4 == 0);
    ggml_cuda_rq4_init_signs();
    constexpr int SUPERBLOCKS_PER_BLOCK = 8;  // 8 warps/block, 256 threads total
    const int64_t nblocks = ne00 / QK_RQ4;
    const dim3 block_dims(32, SUPERBLOCKS_PER_BLOCK, 1);
    const dim3 block_nums(ne10, (nblocks + SUPERBLOCKS_PER_BLOCK - 1) / SUPERBLOCKS_PER_BLOCK,
                          MIN(ne11*ne12, (int64_t) UINT16_MAX));
    const size_t st1 = nb1 / sizeof(dst_t);
    const size_t st2 = nb2 / sizeof(dst_t);
    const size_t st3 = nb3 / sizeof(dst_t);
    const size_t st10 = nb10 / sizeof(int32_t);
    const size_t st11 = nb11 / sizeof(int32_t);
    const size_t st12 = nb12 / sizeof(int32_t);
    k_get_rows_rq4<<<block_nums, block_dims, 0, stream>>>(
        (const block_rq4 *) src0_d, src1_d, dst_d,
        ne00, ne11, ne12,
        st1, st2, st3,
        nb01, nb02, nb03,
        st10, st11, st12);
}

// RQ3 get_rows CUDA kernel: 256-element superblock, 32 lanes/warp, WHT per sub-block.
// 8 sub-blocks of 32; 3-bit level extraction (bits 0-1 in qs, bit 2 in hmask).
template <typename dst_t>
static __global__ void k_get_rows_rq3(
        const block_rq3 * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    constexpr int SUPERBLOCKS_PER_BLOCK = 8;
    const int lane = threadIdx.x;
    const int spw  = threadIdx.y;
    const int64_t z = blockIdx.z;

    if (z >= ne11*ne12) { return; }
    const int64_t nblocks = ne00 / QK_RQ3;
    const int64_t b = (int64_t) blockIdx.y * SUPERBLOCKS_PER_BLOCK + spw;
    if (b >= nblocks) { return; }

    const int i10 = blockIdx.x;
    const int i11 = z / ne12;
    const int i12 = z % ne12;
    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
    const char * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
    const block_rq3 * x = (const block_rq3 *) src0_row + b;
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const float d   = __low2float(x->dm);
    const float dmin = __high2float(x->dm);

    __shared__ float shbuf[SUPERBLOCKS_PER_BLOCK * 32];
    float * sb = shbuf + spw * 32;

    for (int sp = 0; sp < 8; sp++) {
        const int is = sp;
        uint8_t sc, m;
        get_scale_min_k4(is, x->scales, sc, m);
        const float d1 = d * sc; const float m1 = dmin * m;

        const int gi = 32*sp + lane;
        const uint8_t lo = (x->qs[gi/4] >> (2*(gi & 3))) & 3;
        const uint8_t hi = (x->hmask[gi/8] >> (gi & 7)) & 1;
        const uint8_t L = lo | (hi << 2);
        sb[lane] = d1 * L - m1;
        __syncthreads();
        float val = sb[lane];
        for (int step = 1; step < 32; step <<= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, val, step, 32);
            val = (lane & step) ? (other - val) : (other + val);
        }
        val = val * (ggml_cuda_rq3_signs_dev[lane] / sqrtf(32.0f));
        dst_row[b * QK_RQ3 + 32*sp + lane] = ggml_cuda_cast<dst_t>(val);
    }
}

template <typename dst_t>
static void get_rows_cuda_rq3(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_RQ3 == 0);
    ggml_cuda_rq3_init_signs();
    constexpr int SUPERBLOCKS_PER_BLOCK = 8;
    const int64_t nblocks = ne00 / QK_RQ3;
    const dim3 block_dims(32, SUPERBLOCKS_PER_BLOCK, 1);
    const dim3 block_nums(ne10, (nblocks + SUPERBLOCKS_PER_BLOCK - 1) / SUPERBLOCKS_PER_BLOCK,
                          MIN(ne11*ne12, (int64_t) UINT16_MAX));
    const size_t st1 = nb1 / sizeof(dst_t);
    const size_t st2 = nb2 / sizeof(dst_t);
    const size_t st3 = nb3 / sizeof(dst_t);
    const size_t st10 = nb10 / sizeof(int32_t);
    const size_t st11 = nb11 / sizeof(int32_t);
    const size_t st12 = nb12 / sizeof(int32_t);
    k_get_rows_rq3<<<block_nums, block_dims, 0, stream>>>(
        (const block_rq3 *) src0_d, src1_d, dst_d,
        ne00, ne11, ne12,
        st1, st2, st3,
        nb01, nb02, nb03,
        st10, st11, st12);
}
// RQ2 get_rows CUDA kernel: 256-element superblock, 32 lanes/warp, WHT per sub-block.
// 8 sub-blocks of 32; 2-bit level extraction (bits 0-1 in qs). No hmask.
template <typename dst_t>
static __global__ void k_get_rows_rq2(
        const block_rq2 * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    constexpr int SUPERBLOCKS_PER_BLOCK = 8;
    const int lane = threadIdx.x;
    const int spw  = threadIdx.y;
    const int64_t z = blockIdx.z;

    if (z >= ne11*ne12) { return; }
    const int64_t nblocks = ne00 / QK_RQ2;
    const int64_t b = (int64_t) blockIdx.y * SUPERBLOCKS_PER_BLOCK + spw;
    if (b >= nblocks) { return; }

    const int i10 = blockIdx.x;
    const int i11 = z / ne12;
    const int i12 = z % ne12;
    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
    const char * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
    const block_rq2 * x = (const block_rq2 *) src0_row + b;
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const float d   = __low2float(x->dm);
    const float dmin = __high2float(x->dm);

    __shared__ float shbuf[SUPERBLOCKS_PER_BLOCK * 32];
    float * sb = shbuf + spw * 32;

    for (int sp = 0; sp < 8; sp++) {
        const int is = sp;
        uint8_t sc, m;
        get_scale_min_k4(is, x->scales, sc, m);
        const float d1 = d * sc; const float m1 = dmin * m;

        const int gi = 32*sp + lane;
        const uint8_t L = (x->qs[gi/4] >> (2*(gi & 3))) & 3;
        sb[lane] = d1 * L - m1;
        __syncthreads();
        float val = sb[lane];
        for (int step = 1; step < 32; step <<= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, val, step, 32);
            val = (lane & step) ? (other - val) : (other + val);
        }
        val = val * (ggml_cuda_rq2_signs_dev[lane] / sqrtf(32.0f));
        dst_row[b * QK_RQ2 + 32*sp + lane] = ggml_cuda_cast<dst_t>(val);
    }
}

template <typename dst_t>
static void get_rows_cuda_rq2(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_RQ2 == 0);
    ggml_cuda_rq2_init_signs();
    constexpr int SUPERBLOCKS_PER_BLOCK = 8;
    const int64_t nblocks = ne00 / QK_RQ2;
    const dim3 block_dims(32, SUPERBLOCKS_PER_BLOCK, 1);
    const dim3 block_nums(ne10, (nblocks + SUPERBLOCKS_PER_BLOCK - 1) / SUPERBLOCKS_PER_BLOCK,
                          MIN(ne11*ne12, (int64_t) UINT16_MAX));
    const size_t st1 = nb1 / sizeof(dst_t);
    const size_t st2 = nb2 / sizeof(dst_t);
    const size_t st3 = nb3 / sizeof(dst_t);
    const size_t st10 = nb10 / sizeof(int32_t);
    const size_t st11 = nb11 / sizeof(int32_t);
    const size_t st12 = nb12 / sizeof(int32_t);
    k_get_rows_rq2<<<block_nums, block_dims, 0, stream>>>(
        (const block_rq2 *) src0_d, src1_d, dst_d,
        ne00, ne11, ne12,
        st1, st2, st3,
        nb01, nb02, nb03,
        st10, st11, st12);
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
            const int i11 =  dm.x;
            const int i12 =  dm.y;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static __global__ void k_get_rows_kq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    const int64_t nsb = ne00/QK_K; // super-blocks per row
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

        for (int64_t ib = blockIdx.y; ib < nsb; ib += gridDim.y) {
            dequantize_kq(src0_row, ib, dst_row + ib*QK_K, threadIdx.x);
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_lc();
    const src0_t  * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const int32_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * GGML_CUDA_RESTRICT dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const src0_t * GGML_CUDA_RESTRICT src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

template<typename dst_t>
static __global__ void k_get_rows_float_vec(
        const dst_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00v,
        const int64_t ne11, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1_ptr[i10*s10 + i11*s11 + i12*s12];

        int4       * GGML_CUDA_RESTRICT dst_row  = (int4 *)      (dst_ptr + i10*s1 + i11*s2 + i12*s3);
        const int4 * GGML_CUDA_RESTRICT src0_row = (const int4 *)((const char *) src0_ptr + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i = blockIdx.y*blockDim.x + threadIdx.x; i < ne00v; i += gridDim.y*blockDim.x) {
            dst_row[i] = src0_row[i];
        }
    }
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst,
        const int64_t ncols, const int64_t nrows_grad, const int64_t nrows_dst) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    ggml_cuda_pdl_sync();

    // grid.y is clamped to the CUDA grid limit, so stride over the destination rows
    for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
        float sum = 0.0f;

        for (int64_t i = 0; i < nrows_grad; ++i) {
            if (rows[i] != dst_row) {
                continue;
            }
            sum += grad[i*ncols + col];
        }

        dst[dst_row*ncols + col] = sum;
    }
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<int block_dim, typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static void get_rows_cuda_kq(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t nsb = ne00/QK_K;

    const dim3 block_dims(block_dim, 1, 1);
    const dim3 block_nums(ne10, MIN(nsb, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_kq<dst_t, dequantize_kq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    if constexpr (std::is_same<src0_t, dst_t>::value) {
        constexpr int VEC = 16 / sizeof(dst_t);
        const int64_t ne00v = ne00 / VEC;
        const int64_t vec_block_num_y = (ne00v + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
        const bool enough_blocks = vec_block_num_y * ne10 * ne11 * ne12 >= 128;
        const bool can_vec = VEC > 1 && enough_blocks &&
            (ne00 % VEC == 0) &&
            (nb01 % 16 == 0) && (nb02 % 16 == 0) && (nb03 % 16 == 0) &&
            (nb1  % 16 == 0) && (nb2  % 16 == 0) && (nb3  % 16 == 0) &&
            (((uintptr_t) src0_d) % 16 == 0) && (((uintptr_t) dst_d) % 16 == 0);

        if (can_vec) {
            const int block_num_y = vec_block_num_y;
            const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
            ggml_cuda_kernel_launch(k_get_rows_float_vec<dst_t>, launch_params,
                (const dst_t *) src0_d, src1_d, dst_d,
                ne00v, ne11, ne12_fdv,
                s1, s2, s3,
                nb01, nb02, nb03,
                s10, s11, s12);
            return;
        }
    }

    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(k_get_rows_float<src0_t, dst_t>, launch_params,
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q1_0:
            get_rows_cuda_q<QK1_0, QR1_0, dequantize_q1_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_0:
            get_rows_cuda_q<QK2_0, QR2_0, dequantize_q2_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q2_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q3_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q3_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_cuda_kq<32, dst_t, dequantize_q4_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_RQ4:
            get_rows_cuda_rq4(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_RQ3:
            get_rows_cuda_rq3(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_RQ2:
            get_rows_cuda_rq2(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q5_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q6_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_m<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_nl<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_MXFP4:
            get_rows_cuda_kq<32, dst_t, dequantize_mxfp4<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, MIN(ne1, (int64_t)UINT16_MAX), 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10, ne1);
}

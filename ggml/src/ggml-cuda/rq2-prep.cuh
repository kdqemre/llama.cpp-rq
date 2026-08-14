#pragma once

// RQ2 fused activation prep (Kernel A) + batched activation rotation.
//
// Forward randomized Hadamard transform (32-point) on activations so the WHT
// moves off the weight-side load_tiles (plain 2-bit unpack repacked to Q4_K
// nibble layout) onto the activation, amortized once/token across all weight
// rows. See rq2-prep.cu for the design rationale. Mirrors rq4-prep.cuh.

#include "common.cuh"   // pulls ggml-common.h -> block_q8_1, QK8_1; cuda runtime

// RQ2 batched activation rotation for MMQ: fused copy + forward-WHT per
// 32-element sub-block (src->dst), so the standard q8_1 quantizer then produces
// rotated q8_1. Identity: <RHT_inverse(w_rot), a> = <w_rot, RHT_forward(a)>.
void ggml_cuda_rq2_rotate_act(
        float * dst, const float * src, int64_t n, cudaStream_t stream);

// RQ2 fused activation prep (Kernel A): forward-WHT + q8_1 quantize + FP32 Sa
// sidecar. Output q8_1 layout matches quantize_row_q8_1_cuda; ds.x carries the
// rotated activation scale, and `sa_out` carries the ORIGINAL (pre-rotation)
// block sum needed by the RQ2 min term.
void ggml_cuda_rq2_prep_act(
        const float * x, block_q8_1 * vy, float * sa_out,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

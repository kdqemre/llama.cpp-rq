#pragma once

// RQ4 fused activation prep (Kernel A) + batched activation rotation.
//
// Forward randomized Hadamard transform (32-point) on activations so the WHT
// moves off the weight-side load_tiles (which becomes a plain Q4_K nibble-unpack)
// onto the activation, amortized once/token across all weight rows. See rq4-prep.cu
// for the full design rationale. Extracted from tq3-native.cu/.cuh in the source
// repo (minus the unrelated tq3_0 code) so the live 11-arg prep API + FP32 Sa
// sidecar matches what the RQ4 MMVQ/MMQ dispatch calls.

#include "common.cuh"   // pulls ggml-common.h -> block_q8_1, QK8_1; cuda runtime

// RQ4 batched activation rotation for MMQ: fused copy + forward-WHT per
// 32-element sub-block (src->dst), so the standard q8_1 quantizer then produces
// rotated q8_1. Identity: <RHT_inverse(w_rot), a> = <w_rot, RHT_forward(a)>.
void ggml_cuda_rq4_rotate_act(
        float * dst, const float * src, int64_t n, cudaStream_t stream);

// RQ4 fused activation prep (Kernel A): forward-WHT + q8_1 quantize + FP32 Sa
// sidecar. Output q8_1 layout matches quantize_row_q8_1_cuda; ds.x carries the
// rotated activation scale, and `sa_out` carries the ORIGINAL (pre-rotation)
// block sum needed by the RQ4 min term. Replaces memcpy+rotate+quantize+fixup.
void ggml_cuda_rq4_prep_act(
        const float * x, block_q8_1 * vy, float * sa_out,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

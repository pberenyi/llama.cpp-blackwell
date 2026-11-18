#pragma once

#include "common.cuh"

// FlashAttention for large batches with q4_0 or F16 K/V and head size 256 (e.g. Qwen 3.5/3.6/3.8 prompt processing):
//     K*Q^T is computed with int8 tensor cores (K requantized to int8 per row), V*softmax with FP16 tensor cores.
bool ggml_cuda_flash_attn_ext_mma_q4i8_supported(const int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_mma_q4i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

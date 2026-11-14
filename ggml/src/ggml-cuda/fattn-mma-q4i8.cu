// FlashAttention for large batches (prompt processing) with q4_0 or F16 K/V, head size 256 and GQA,
//     INT8 tensor cores for K*Q^T, FP16 tensor cores for V*softmax. Only used on sm_120 (see ..._supported).
//
// Q is quantized to int8 once per tile with one scale per column (the softmax scale and log2(e) are folded in).
// Each K tile is requantized in shared memory to int8 with a single scale per row:
//     q4_0: K'[i] = round((q[i] - 8) * d[b(i)] / s) with s = max|d| * 8/127. For Hadamard rotated K (attention rotation)
//     the block scales of a row are similar and the added rounding error is ~1% of the q4_0 error.
//     F16:  K'[i] = round(K[i] / s) with s = max|K| / 127.
//     With a single scale per row the int32 dot products of all 256 values accumulate in the tensor cores,
//     the int32 accumulators are initialized with the bit pattern of 1.5*2^23 so that they can be converted to float
//     with a single FADD (|dot| <= 256*127*127 < 2^22).
// Softmax is done in FP32 (log2 domain), V*softmax uses FP16 tensor cores with FP16 accumulation like the MMA kernel
//     in fattn-mma-f16.cuh (q4_0 V is dequantized to FP16 in shared memory, F16 V is copied there directly).
// Pipeline: the tiles converted for the tensor cores are double buffered, iteration i computes tile i, converts tile i + 1
//     and loads the raw data of tile i + 2 with cp.async, with a single __syncthreads per iteration.
//     The first half of the warps converts after its tensor core work, the second half before it, so that the ALU work
//     of one half overlaps with the MMAs of the other half.
// The work is distributed with the same stream-k scheme as the MMA kernel so that its fixup kernels can be reused.

#include "common.cuh"
#include "cp-async.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-q4i8.cuh"

// Defined in fattn.cu: whether the q4_0 K/V data can be read directly (the F16 K/V buffers are then not allocated).
bool ggml_cuda_flash_attn_ext_mma_f16_use_q4_0_KV(const int cc, const ggml_tensor * dst);

namespace {

constexpr int   Q4I8_D         = 256;                              // head size of K, Q and V
constexpr int   Q4I8_NBLK      = Q4I8_D/QK4_0;                     // q4_0 blocks per K/V row
constexpr int   Q4I8_ROW_RAW   = Q4I8_NBLK*sizeof(block_q4_0);     // bytes per K/V row
constexpr int   Q4I8_NBATCH    = 32;                               // K/V rows per iteration
constexpr int   Q4I8_NWARPS    = 8;
constexpr int   Q4I8_NTHREADS  = Q4I8_NWARPS*WARP_SIZE;
constexpr int   Q4I8_NCOLS     = 16*Q4I8_NWARPS;                   // Q columns (tokens x heads) per CUDA block
constexpr int   Q4I8_VH_STRIDE = Q4I8_D + 8;                       // halves per row of the dequantized V tile

// Shared memory layout in bytes. The tiles converted for the tensor cores are double buffered so that the next tile
//     can be converted while the current one is being used, the raw K/V tiles are loaded 2 iterations ahead.
constexpr int   Q4I8_OFF_KQ    = 0;                                            // K as int8, [row][256 B], 16 B chunks swizzled
constexpr int   Q4I8_OFF_SK    = Q4I8_OFF_KQ + Q4I8_NBATCH*Q4I8_D;             // K row scales as float
constexpr int   Q4I8_OFF_VH    = Q4I8_OFF_SK + Q4I8_NBATCH*4;                  // V dequantized to FP16, [row][Q4I8_VH_STRIDE]
constexpr int   Q4I8_CONV_SIZE = Q4I8_OFF_VH + Q4I8_NBATCH*Q4I8_VH_STRIDE*2;   // offsets above are relative to a conversion buffer
constexpr int   Q4I8_RAW_BYTES = Q4I8_NBATCH*Q4I8_ROW_RAW;
constexpr int   Q4I8_OFF_RAW   = 2*Q4I8_CONV_SIZE;                             // 2 stages of raw K + raw V (q4_0) or raw K (F16)

// Q staging at the beginning of a tile, overlaps everything else:
constexpr int   Q4I8_QS_STRIDE = Q4I8_D + 16;                                  // bytes per int8 Q column
constexpr int   Q4I8_OFF_QS    = 0;
constexpr int   Q4I8_OFF_DQ    = Q4I8_OFF_QS + Q4I8_NCOLS*Q4I8_QS_STRIDE;

constexpr int   Q4I8_MAGIC_I   = 0x4B400000;                        // bit pattern of 1.5*2^23
constexpr float Q4I8_MAGIC_F   = 12582912.0f;
constexpr float Q4I8_LOG2E     = 1.4426950408889634f;
constexpr float Q4I8_LN2       = 0.6931471805599453f;

constexpr __host__ __device__ int q4i8_mask_bytes(const int ncols1) {
    return ncols1*Q4I8_NBATCH*(int) sizeof(half);
}

// Bytes per stage of raw data: K + V for q4_0, only K for F16 (V is copied directly to the FP16 V tile).
constexpr __host__ __device__ int q4i8_raw_bytes(const bool kv_f16) {
    return kv_f16 ? Q4I8_NBATCH*Q4I8_D*(int) sizeof(half) : 2*Q4I8_RAW_BYTES;
}

constexpr __host__ __device__ int q4i8_off_mask(const bool kv_f16) {
    return Q4I8_OFF_RAW + 2*q4i8_raw_bytes(kv_f16); // 2 stages of mask tiles
}

constexpr __host__ __device__ int q4i8_smem_bytes(const int ncols1, const bool kv_f16) {
    return q4i8_off_mask(kv_f16) + 2*q4i8_mask_bytes(ncols1);
}

// Physical 16 byte chunk of the logical chunk c (8 halves) in a row of the dequantized V tile.
// The XOR spreads the stores of the 4 threads per row over different banks, the ldmatrix loads stay conflict-free.
static __device__ __forceinline__ int q4i8_vh_chunk(const int c) {
    return c ^ (2*((c/8) % 4));
}

static __device__ __forceinline__ float q4i8_exp2(const float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ void q4i8_ldmatrix_x4(uint32_t (&r)[4], const uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

static __device__ __forceinline__ void q4i8_ldmatrix_x4_trans(uint32_t (&r)[4], const uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// D += A*B, A: 16x32 s8 row-major (Q), B: 32x8 s8 column-major (K).
static __device__ __forceinline__ void q4i8_mma_s8(int (&d)[4], const int (&a)[4], const uint32_t b0, const uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// D += A*B, A: 16x16 f16 row-major (softmax), B: 16x8 f16 column-major (V), FP16 accumulation.
static __device__ __forceinline__ void q4i8_mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// F16 V rows are copied directly into the FP16 V tile of a conversion buffer.
static __device__ __forceinline__ void q4i8_load_V_f16(const char * GGML_CUDA_RESTRICT V, char * GGML_CUDA_RESTRICT conv, const int nb21, const int k_VKQ_0) {
    constexpr int chunks_per_row = Q4I8_D*sizeof(half)/16;
    constexpr int chunks         = Q4I8_NBATCH*chunks_per_row;
    static_assert(chunks % Q4I8_NTHREADS == 0, "bad thread count");

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const unsigned int Vh_32 = ggml_cuda_cvta_generic_to_shared(conv + Q4I8_OFF_VH);

#pragma unroll
    for (int c0 = 0; c0 < chunks; c0 += Q4I8_NTHREADS) {
        const int c = c0 + tid;
        const int i = c / chunks_per_row;
        const int k = c % chunks_per_row;
        cp_async_cg_16<64>(Vh_32 + i*(Q4I8_VH_STRIDE*sizeof(half)) + 16*q4i8_vh_chunk(k), V + int64_t(k_VKQ_0 + i)*nb21 + 16*k);
    }
}

template <bool kv_f16>
static __device__ __forceinline__ void q4i8_load_KV(
        const char * GGML_CUDA_RESTRICT K, const char * GGML_CUDA_RESTRICT V, char * GGML_CUDA_RESTRICT stage, const int nb11, const int nb21, const int k_VKQ_0) {
    if constexpr (kv_f16) {
        constexpr int chunks_per_row = Q4I8_D*sizeof(half)/16;
        constexpr int chunks         = Q4I8_NBATCH*chunks_per_row;
        static_assert(chunks % Q4I8_NTHREADS == 0, "bad thread count");

        const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
        const unsigned int stage_32 = ggml_cuda_cvta_generic_to_shared(stage);

#pragma unroll
        for (int c0 = 0; c0 < chunks; c0 += Q4I8_NTHREADS) {
            const int c = c0 + tid;
            const int i = c / chunks_per_row;
            const int k = c % chunks_per_row;
            cp_async_cg_16<64>(stage_32 + 16*c, K + int64_t(k_VKQ_0 + i)*nb11 + 16*k);
        }
        GGML_UNUSED_VARS(V, nb21);
        return;
    }

    constexpr int chunks_per_row = Q4I8_ROW_RAW/16;
    constexpr int chunks_KV      = Q4I8_NBATCH*chunks_per_row;
    static_assert(Q4I8_ROW_RAW % 16 == 0, "bad row size");

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const unsigned int stage_32 = ggml_cuda_cvta_generic_to_shared(stage);

#pragma unroll
    for (int c0 = 0; c0 < chunks_KV; c0 += Q4I8_NTHREADS) {
        const int c = c0 + tid;
        if (c0 + Q4I8_NTHREADS > chunks_KV && c >= chunks_KV) {
            break;
        }
        const int i = c / chunks_per_row;
        const int k = c % chunks_per_row;
        cp_async_cg_16<64>(stage_32 +                  16*c, K + int64_t(k_VKQ_0 + i)*nb11 + 16*k);
        cp_async_cg_16<64>(stage_32 + Q4I8_RAW_BYTES + 16*c, V + int64_t(k_VKQ_0 + i)*nb21 + 16*k);
    }
}

template <int ncols1>
static __device__ __forceinline__ void q4i8_load_mask(
        const half * GGML_CUDA_RESTRICT mask_h, char * GGML_CUDA_RESTRICT tile_mask, const int stride_mask, const int k_VKQ_0, const int jt, const uint3 ne01) {
    constexpr int chunks_mask = q4i8_mask_bytes(ncols1)/16;

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
    for (int c0 = 0; c0 < chunks_mask; c0 += Q4I8_NTHREADS) {
        const int c = c0 + tid;
        if (c0 + Q4I8_NTHREADS > chunks_mask && c >= chunks_mask) {
            break;
        }
        constexpr int chunks_per_mask_row = Q4I8_NBATCH*sizeof(half)/16;
        const int j      = c / chunks_per_mask_row;
        const int k      = c % chunks_per_mask_row;
        const int j_vram = fastmodulo(jt*ncols1 + j, ne01);
        cp_async_cg_16<0>(tile_mask_32 + 16*c, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 8*k);
    }
}

// F16 K: requantize to int8 with one scale per row. The first half of the warps converts the first half of the rows.
static __device__ __forceinline__ void q4i8_convert_K_f16(const char * GGML_CUDA_RESTRICT stage, char * GGML_CUDA_RESTRICT conv) {
    static_assert(Q4I8_NTHREADS == 256, "bad thread count");
    constexpr int seg_per_row = 8; // 32 values per thread

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int r   = (Q4I8_NBATCH/2)*(tid / (Q4I8_NTHREADS/2)) + (tid % (Q4I8_NTHREADS/2)) / seg_per_row;
    const int seg = tid % seg_per_row;

    const uint4 * src = (const uint4 *) (stage + r*(Q4I8_D*sizeof(half)) + seg*(Q4I8_D*sizeof(half)/seg_per_row));
    uint4 raw[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        raw[l] = src[l];
    }
    const half2 * h2 = (const half2 *) raw;

    half2 amax2 = __habs2(h2[0]);
#pragma unroll
    for (int l = 1; l < 16; ++l) {
        amax2 = __hmax2(amax2, __habs2(h2[l]));
    }
    float amax = fmaxf(__low2float(amax2), __high2float(amax2));
    // The 8 threads of a row are consecutive lanes.
#pragma unroll
    for (int offset = 1; offset < seg_per_row; offset <<= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    const float inv = amax > 0.0f ? 127.0f/amax : 0.0f;
    if (seg == 0) {
        ((float *) (conv + Q4I8_OFF_SK))[r] = amax/127.0f;
    }

    // MAGIC_F + x is rounded to an integer with the int8 value x in the low byte.
    uint32_t out[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const float2 f0 = __half22float2(h2[2*l + 0]);
        const float2 f1 = __half22float2(h2[2*l + 1]);
        const uint32_t i0 = __float_as_uint(fmaf(f0.x, inv, Q4I8_MAGIC_F));
        const uint32_t i1 = __float_as_uint(fmaf(f0.y, inv, Q4I8_MAGIC_F));
        const uint32_t i2 = __float_as_uint(fmaf(f1.x, inv, Q4I8_MAGIC_F));
        const uint32_t i3 = __float_as_uint(fmaf(f1.y, inv, Q4I8_MAGIC_F));
        out[l] = __byte_perm(__byte_perm(i0, i1, 0x0040), __byte_perm(i2, i3, 0x0040), 0x5410);
    }
    char * dst_row = conv + Q4I8_OFF_KQ + r*Q4I8_D;
    *(int4 *) (dst_row + 16*((2*seg + 0) ^ (r % 8))) = make_int4(out[0], out[1], out[2], out[3]);
    *(int4 *) (dst_row + 16*((2*seg + 1) ^ (r % 8))) = make_int4(out[4], out[5], out[6], out[7]);
}

// Requantize the raw K tile to int8 with one scale per row, dequantize the raw V tile to FP16 (only for q4_0).
// For q4_0 each thread handles 2 consecutive q4_0 blocks of one K row and/or of one V row.
template <bool kv_f16>
static __device__ __forceinline__ void q4i8_convert_stage(const char * GGML_CUDA_RESTRICT stage, char * GGML_CUDA_RESTRICT conv) {
    if constexpr (kv_f16) {
        q4i8_convert_K_f16(stage, conv);
        return;
    }

    constexpr int npairs = Q4I8_NBATCH*Q4I8_NBLK/2;
    static_assert(Q4I8_NTHREADS == 2*npairs, "bad thread count");

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int r   = (tid % npairs) / (Q4I8_NBLK/2);
    const int bp  = (tid % npairs) % (Q4I8_NBLK/2);

    // The first half of the warps converts K, the second half V.
    const bool convert_K = tid <  npairs;
    const bool convert_V = tid >= npairs;

    // 0x64 in the high byte of a half turns a nibble n into 1024 + n, subtracting 1032 gives n - 8 exactly.
    const half2 offset = make_half2(1032.0f, 1032.0f);

    if (convert_K) {
        // Two consecutive 18 byte blocks start at a 4 byte aligned offset.
        const uint32_t * src = (const uint32_t *) (stage + r*Q4I8_ROW_RAW + bp*2*sizeof(block_q4_0));
        uint32_t w[9];
#pragma unroll
        for (int l = 0; l < 9; ++l) {
            w[l] = src[l];
        }
        const float d0 = __half2float(__ushort_as_half(w[0] & 0xFFFF));
        const float d1 = __half2float(__ushort_as_half(w[4] >> 16));

        // The 4 threads of a row are consecutive lanes.
        float amax = fmaxf(fabsf(d0), fabsf(d1));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1, WARP_SIZE));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2, WARP_SIZE));
        const float inv = amax > 0.0f ? 127.0f/(8.0f*amax) : 0.0f;
        if (bp == 0) {
            ((float *) (conv + Q4I8_OFF_SK))[r] = amax*(8.0f/127.0f);
        }

#pragma unroll
        for (int ib = 0; ib < 2; ++ib) {
            const half2 ratio = __float2half2_rn((ib == 0 ? d0 : d1)*inv); // |ratio| <= 127/8
            uint32_t out[8];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t qs = ib == 0 ? __funnelshift_r(w[l], w[l + 1], 16) : w[5 + l];
                const uint32_t lo =  qs       & 0x0F0F0F0F;
                const uint32_t hi = (qs >> 4) & 0x0F0F0F0F;
                const uint32_t tmp[4] = {
                    __byte_perm(lo, 0x64646464, 0x4140), __byte_perm(lo, 0x64646464, 0x4342),
                    __byte_perm(hi, 0x64646464, 0x4140), __byte_perm(hi, 0x64646464, 0x4342),
                };
                // 1536 + x is rounded to an integer with the int8 value x in the low byte of the half.
                uint32_t h[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const half2 v = __hfma2(*((const half2 *) &tmp[m]) - offset, ratio, make_half2(1536.0f, 1536.0f));
                    h[m] = *(const uint32_t *) &v;
                }
                out[l + 0] = __byte_perm(h[0], h[1], 0x6420); // values 4*l + 0..3
                out[l + 4] = __byte_perm(h[2], h[3], 0x6420); // values 4*l + 16..19
            }
            const int b = 2*bp + ib;
            char * dst_row = conv + Q4I8_OFF_KQ + r*Q4I8_D;
            *(int4 *) (dst_row + 16*((2*b + 0) ^ (r % 8))) = make_int4(out[0], out[1], out[2], out[3]);
            *(int4 *) (dst_row + 16*((2*b + 1) ^ (r % 8))) = make_int4(out[4], out[5], out[6], out[7]);
        }
    }

    if (convert_V) {
        const uint32_t * src = (const uint32_t *) (stage + Q4I8_RAW_BYTES + r*Q4I8_ROW_RAW + bp*2*sizeof(block_q4_0));
        uint32_t w[9];
#pragma unroll
        for (int l = 0; l < 9; ++l) {
            w[l] = src[l];
        }

        half * Vh = (half *) (conv + Q4I8_OFF_VH) + r*Q4I8_VH_STRIDE;

#pragma unroll
        for (int ib = 0; ib < 2; ++ib) {
            uint32_t qs[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                qs[l] = ib == 0 ? __funnelshift_r(w[l], w[l + 1], 16) : w[5 + l];
            }
            const half2 d = __half2half2(__ushort_as_half(ib == 0 ? w[0] & 0xFFFF : w[4] >> 16));

            half2 vals[QK4_0/2];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t lo =  qs[l]       & 0x0F0F0F0F;
                const uint32_t hi = (qs[l] >> 4) & 0x0F0F0F0F;
                const uint32_t tmp[4] = {
                    __byte_perm(lo, 0x64646464, 0x4140), __byte_perm(lo, 0x64646464, 0x4342),
                    __byte_perm(hi, 0x64646464, 0x4140), __byte_perm(hi, 0x64646464, 0x4342),
                };
                vals[2*l + 0]           = (*((const half2 *) &tmp[0]) - offset) * d;
                vals[2*l + 1]           = (*((const half2 *) &tmp[1]) - offset) * d;
                vals[QK4_0/4 + 2*l + 0] = (*((const half2 *) &tmp[2]) - offset) * d;
                vals[QK4_0/4 + 2*l + 1] = (*((const half2 *) &tmp[3]) - offset) * d;
            }

#pragma unroll
            for (int c = 0; c < QK4_0/8; ++c) {
                *(int4 *) (Vh + 8*q4i8_vh_chunk((2*bp + ib)*(QK4_0/8) + c)) = *(const int4 *) (vals + 4*c);
            }
        }
    }
}

template <int ncols1, int ncols2, bool kv_f16, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void q4i8_process_tile(
        const float2 * const GGML_CUDA_RESTRICT Q_f2,
        const char   * const GGML_CUDA_RESTRICT K,
        const char   * const GGML_CUDA_RESTRICT V,
        const half   * const GGML_CUDA_RESTRICT mask_h,
        float2       * const GGML_CUDA_RESTRICT dstk,
        float2       * const GGML_CUDA_RESTRICT dst_meta,
        const float scale,
        const uint3 ne01,
        const int ne02,
        const int stride_Q1,
        const int stride_Q2,
        const int nb11,
        const int nb21,
        const int stride_mask,
        const int jt,
        const int kb0_start,
        const int kb0_stop) {
    constexpr int ncols = ncols1*ncols2;
    static_assert(ncols == Q4I8_NCOLS, "bad ncols");
    static_assert(Q4I8_OFF_DQ + Q4I8_NCOLS*4 <= q4i8_smem_bytes(ncols1, kv_f16), "Q staging does not fit");
    constexpr int raw_bytes = q4i8_raw_bytes(kv_f16);
    constexpr int off_mask  = q4i8_off_mask(kv_f16);

    extern __shared__ __align__(16) char smem[];

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int t    = lane % 4;

    // Quantize Q to int8, one scale per column. The softmax scale and log2(e) are folded into the Q scale.
    {
        const float qscale = scale*Q4I8_LOG2E;
#pragma unroll 2
        for (int jc0 = 0; jc0 < 16; ++jc0) {
            const int jc = 16*warp + jc0;
            const int j  = jc / ncols2;
            const int c  = jc % ncols2;

            float x[8];
            if (ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) {
                const float4 * src = (const float4 *) (Q_f2 + (jt*ncols1 + j)*stride_Q1 + c*stride_Q2) + 2*lane;
                const float4 v0 = src[0];
                const float4 v1 = src[1];
                x[0] = v0.x; x[1] = v0.y; x[2] = v0.z; x[3] = v0.w;
                x[4] = v1.x; x[5] = v1.y; x[6] = v1.z; x[7] = v1.w;
            } else {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    x[l] = 0.0f;
                }
            }

            float amax = 0.0f;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                x[l] *= qscale;
                amax = fmaxf(amax, fabsf(x[l]));
            }
            amax = warp_reduce_max<WARP_SIZE>(amax);
            const float d  = amax / 127.0f;
            const float id = amax > 0.0f ? 127.0f/amax : 0.0f;

            int q[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                q[l] = __float2int_rn(x[l]*id);
            }
            const int w0 = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | (q[3] << 24);
            const int w1 = (q[4] & 0xFF) | ((q[5] & 0xFF) << 8) | ((q[6] & 0xFF) << 16) | (q[7] << 24);
            *(int2 *) (smem + Q4I8_OFF_QS + jc*Q4I8_QS_STRIDE + 8*lane) = make_int2(w0, w1);
            if (lane == 0) {
                ((float *) (smem + Q4I8_OFF_DQ))[jc] = d;
            }
        }
    }
    __syncthreads();

    const int jc_lo = 16*warp + g;
    const int jc_hi = jc_lo + 8;

    int   Q_A[Q4I8_NBLK][4];   // int8 Q fragments for mma.m16n8k32
    float dQ[2];
#pragma unroll
    for (int b = 0; b < Q4I8_NBLK; ++b) {
        const int * Qs = (const int *) (smem + Q4I8_OFF_QS);
        Q_A[b][0] = Qs[jc_lo*(Q4I8_QS_STRIDE/4) + 8*b + t];
        Q_A[b][1] = Qs[jc_hi*(Q4I8_QS_STRIDE/4) + 8*b + t];
        Q_A[b][2] = Qs[jc_lo*(Q4I8_QS_STRIDE/4) + 8*b + 4 + t];
        Q_A[b][3] = Qs[jc_hi*(Q4I8_QS_STRIDE/4) + 8*b + 4 + t];
    }
    dQ[0] = ((const float *) (smem + Q4I8_OFF_DQ))[jc_lo];
    dQ[1] = ((const float *) (smem + Q4I8_OFF_DQ))[jc_hi];

    // Mask rows of this thread's columns:
    const int j_lo = jc_lo / ncols2;
    const int j_hi = jc_hi / ncols2;

    uint32_t VKQ[Q4I8_D/8][2] = {{0}}; // FP16 accumulators, [8 dims][column g, column g + 8]
    float KQ_max[2]    = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};
    float KQ_rowsum[2] = {0.0f, 0.0f};

    const int kb0_stop_eff = max(kb0_stop, kb0_start + 1); // Positions beyond KV_max are masked, always do >= 1 iteration.

    __syncthreads(); // Q staging is overwritten by the K/V tiles.

    // Pipeline: iteration i uses the converted tile i, converts the raw tile i + 1, and loads the raw tile i + 2 + mask i + 1.
    q4i8_load_KV<kv_f16>(K, V, smem + Q4I8_OFF_RAW, nb11, nb21, kb0_start*Q4I8_NBATCH);
    if constexpr (kv_f16) {
        q4i8_load_V_f16(V, smem, nb21, kb0_start*Q4I8_NBATCH);
    }
    q4i8_load_mask<ncols1>(mask_h, smem + off_mask, stride_mask, kb0_start*Q4I8_NBATCH, jt, ne01);
    if (kb0_start + 1 < kb0_stop_eff) {
        q4i8_load_KV<kv_f16>(K, V, smem + Q4I8_OFF_RAW + raw_bytes, nb11, nb21, (kb0_start + 1)*Q4I8_NBATCH);
    }
    cp_async_wait_all();
    __syncthreads();
    q4i8_convert_stage<kv_f16>(smem + Q4I8_OFF_RAW, smem);

    const uint32_t smem_32 = ggml_cuda_cvta_generic_to_shared(smem);
    const bool convert_first = threadIdx.y >= Q4I8_NWARPS/2;

    for (int kb0 = kb0_start; kb0 < kb0_stop_eff; ++kb0) {
        const int  it       = kb0 - kb0_start;
        const int  off_conv = (it % 2)*Q4I8_CONV_SIZE;
        const char * conv   = smem + off_conv;
        const char * tile_mask = smem + off_mask + (it % 2)*q4i8_mask_bytes(ncols1);

        cp_async_wait_all();
        __syncthreads();
        if (kb0 + 2 < kb0_stop_eff) {
            q4i8_load_KV<kv_f16>(K, V, smem + Q4I8_OFF_RAW + (it % 2)*raw_bytes, nb11, nb21, (kb0 + 2)*Q4I8_NBATCH);
        }
        if (kb0 + 1 < kb0_stop_eff) {
            q4i8_load_mask<ncols1>(mask_h, smem + off_mask + ((it + 1) % 2)*q4i8_mask_bytes(ncols1), stride_mask,
                (kb0 + 1)*Q4I8_NBATCH, jt, ne01);
            if constexpr (kv_f16) {
                q4i8_load_V_f16(V, smem + ((it + 1) % 2)*Q4I8_CONV_SIZE, nb21, (kb0 + 1)*Q4I8_NBATCH);
            }
        }
        if (convert_first && kb0 + 1 < kb0_stop_eff) {
            q4i8_convert_stage<kv_f16>(smem + Q4I8_OFF_RAW + ((it + 1) % 2)*raw_bytes, smem + ((it + 1) % 2)*Q4I8_CONV_SIZE);
        }

        // KQ = K*Q^T, 4 tiles of 8 K rows x 16 columns, all 256 values accumulated in int32:
        int KQ_i[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                KQ_i[nt][l] = Q4I8_MAGIC_I;
            }
        }
#pragma unroll
        for (int kc = 0; kc < Q4I8_NBLK; ++kc) {
#pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2) {
                // Matrices: rows 16*nt2 + 0..7 / 8..15, chunks 2*kc / 2*kc + 1.
                const int row   = 16*nt2 + 8*(lane / 16) + lane % 8;
                const int chunk = 2*kc + (lane / 8) % 2;
                uint32_t K_B[4];
                q4i8_ldmatrix_x4(K_B, smem_32 + off_conv + Q4I8_OFF_KQ + row*Q4I8_D + 16*(chunk ^ (lane % 8)));
                q4i8_mma_s8(KQ_i[2*nt2 + 0], Q_A[kc], K_B[0], K_B[1]);
                q4i8_mma_s8(KQ_i[2*nt2 + 1], Q_A[kc], K_B[2], K_B[3]);
            }
        }

        // Scale, mask, softmax in the log2 domain:
        const half2  * mask_lo = (const half2 *) (tile_mask + j_lo*Q4I8_NBATCH*sizeof(half));
        const half2  * mask_hi = (const half2 *) (tile_mask + j_hi*Q4I8_NBATCH*sizeof(half));
        const float2 * sK      = (const float2 *) (conv + Q4I8_OFF_SK);
        float KQ[4][4];
        float KQ_max_new[2] = {KQ_max[0], KQ_max[1]};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float2 m_lo = __half22float2(mask_lo[4*nt + t]);
            const float2 m_hi = __half22float2(mask_hi[4*nt + t]);
            const float2 s    = sK[4*nt + t];
            KQ[nt][0] = fmaf((__int_as_float(KQ_i[nt][0]) - Q4I8_MAGIC_F)*s.x, dQ[0], Q4I8_LOG2E*m_lo.x);
            KQ[nt][1] = fmaf((__int_as_float(KQ_i[nt][1]) - Q4I8_MAGIC_F)*s.y, dQ[0], Q4I8_LOG2E*m_lo.y);
            KQ[nt][2] = fmaf((__int_as_float(KQ_i[nt][2]) - Q4I8_MAGIC_F)*s.x, dQ[1], Q4I8_LOG2E*m_hi.x);
            KQ[nt][3] = fmaf((__int_as_float(KQ_i[nt][3]) - Q4I8_MAGIC_F)*s.y, dQ[1], Q4I8_LOG2E*m_hi.y);
            // FATTN_KQ_MAX_OFFSET is 3*ln(2), i.e. 3 in the log2 domain.
            KQ_max_new[0] = fmaxf(KQ_max_new[0], fmaxf(KQ[nt][0], KQ[nt][1]) + 3.0f);
            KQ_max_new[1] = fmaxf(KQ_max_new[1], fmaxf(KQ[nt][2], KQ[nt][3]) + 3.0f);
        }
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], 1, WARP_SIZE));
            KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], 2, WARP_SIZE));
        }

        float KQ_max_scale[2];
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            KQ_max_scale[col] = q4i8_exp2(KQ_max[col] - KQ_max_new[col]);
            KQ_max[col]       = KQ_max_new[col];
            KQ_rowsum[col]   *= KQ_max_scale[col];
        }

        uint32_t P_A[2][4]; // softmax as FP16 A fragments for mma.m16n8k16, [16 K rows]
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p0 = q4i8_exp2(KQ[nt][0] - KQ_max_new[0]);
            const float p1 = q4i8_exp2(KQ[nt][1] - KQ_max_new[0]);
            const float p2 = q4i8_exp2(KQ[nt][2] - KQ_max_new[1]);
            const float p3 = q4i8_exp2(KQ[nt][3] - KQ_max_new[1]);
            KQ_rowsum[0] += p0 + p1;
            KQ_rowsum[1] += p2 + p3;
            const half2 p01 = __floats2half2_rn(p0, p1);
            const half2 p23 = __floats2half2_rn(p2, p3);
            P_A[nt/2][2*(nt % 2) + 0] = *(const uint32_t *) &p01;
            P_A[nt/2][2*(nt % 2) + 1] = *(const uint32_t *) &p23;
        }

        // Rescaling the accumulators is only necessary if the maximum changed for at least one column of the warp:
        if (__any_sync(0xFFFFFFFF, KQ_max_scale[0] != 1.0f || KQ_max_scale[1] != 1.0f)) {
            const half2 s_lo = __float2half2_rn(KQ_max_scale[0]);
            const half2 s_hi = __float2half2_rn(KQ_max_scale[1]);
#pragma unroll
            for (int dn = 0; dn < Q4I8_D/8; ++dn) {
                *(half2 *) &VKQ[dn][0] *= s_lo;
                *(half2 *) &VKQ[dn][1] *= s_hi;
            }
        }

        // VKQ += softmax*V:
#pragma unroll
        for (int ks = 0; ks < Q4I8_NBATCH/16; ++ks) {
#pragma unroll
            for (int dn2 = 0; dn2 < Q4I8_D/16; ++dn2) {
                const int row = 16*ks + lane % 8 + 8*((lane / 8) % 2);
                const int col = 8*q4i8_vh_chunk(2*dn2 + lane / 16);
                uint32_t V_B[4];
                q4i8_ldmatrix_x4_trans(V_B, smem_32 + off_conv + Q4I8_OFF_VH + (row*Q4I8_VH_STRIDE + col)*sizeof(half));
                q4i8_mma_f16(VKQ[2*dn2 + 0], P_A[ks], V_B[0], V_B[1]);
                q4i8_mma_f16(VKQ[2*dn2 + 1], P_A[ks], V_B[2], V_B[3]);
            }
        }
        // Warps 0..NWARPS/2-1 convert K after their tensor core work, the other warps convert V before it.
        //     The two groups are thereby out of phase and the ALU work of one group overlaps with the MMAs of the other.
        if (!convert_first && kb0 + 1 < kb0_stop_eff) {
            q4i8_convert_stage<kv_f16>(smem + Q4I8_OFF_RAW + ((it + 1) % 2)*raw_bytes, smem + ((it + 1) % 2)*Q4I8_CONV_SIZE);
        }
    }

#pragma unroll
    for (int col = 0; col < 2; ++col) {
        KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], 1, WARP_SIZE);
        KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], 2, WARP_SIZE);
    }

    // Write back results. The fixup kernels expect the KQ maxima in the natural log domain.
    if ((needs_fixup || is_fixup) && t == 0) {
        float2 * meta = dst_meta + (is_fixup ? gridDim.x + blockIdx.x : blockIdx.x)*ncols;
        meta[jc_lo] = make_float2(KQ_max[0]*Q4I8_LN2, KQ_rowsum[0]);
        meta[jc_hi] = make_float2(KQ_max[1]*Q4I8_LN2, KQ_rowsum[1]);
    }

#pragma unroll
    for (int col = 0; col < 2; ++col) {
        const int jc = col == 0 ? jc_lo : jc_hi;
        const int j  = jc / ncols2;
        const int c  = jc % ncols2;

        float2 * dst_col;
        if (is_fixup) {
            dst_col = dst_meta + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(Q4I8_D/2)) + jc*(Q4I8_D/2);
        } else {
            if (ncols1 > 1 && jt*ncols1 + j >= int(ne01.z)) {
                continue;
            }
            dst_col = dstk + ((jt*ncols1 + j)*ne02 + c)*(Q4I8_D/2);
        }
        const float inv_rowsum = needs_fixup || is_fixup ? 1.0f : 1.0f/KQ_rowsum[col];
#pragma unroll
        for (int dn = 0; dn < Q4I8_D/8; ++dn) {
            const float2 v = __half22float2(*(const half2 *) &VKQ[dn][col]);
            dst_col[4*dn + t] = make_float2(v.x*inv_rowsum, v.y*inv_rowsum);
        }
    }
}

template <int ncols1, int ncols2, bool kv_f16>
__launch_bounds__(Q4I8_NTHREADS, 1)
static __global__ void flash_attn_ext_q4i8(
        const char * Q,
        const char * K,
        const char * V,
        const char * mask,
        const char * sinks,
        const int  * KV_max,
        float      * dst,
        float2     * dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_sync();
#if defined(FLASH_ATTN_AVAILABLE) && defined(BLACKWELL_MMA_AVAILABLE)
    constexpr int nbatch_fa = Q4I8_NBATCH;

    const int gqa_ratio = ne02 / ne12;

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_mask = nb31 / sizeof(half);

    const int iter_k     = ne11 / nbatch_fa;
    const int iter_j     = (ne01.z + (ncols1 - 1)) / ncols1;
    const int iter_z_gqa = gqa_ratio / ncols2;

    // kbc == k block continuous, current index in continuous ijk space. Same work distribution as flash_attn_ext_f16.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    auto tile = [&](auto needs_fixup_c, auto is_fixup_c) {
        constexpr bool needs_fixup = decltype(needs_fixup_c)::value;
        constexpr bool is_fixup    = decltype(is_fixup_c)::value;

        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const char   * K_z    = K + nb13*sequence + nb12*z_KV;
        const char   * V_z    = V + nb23*sequence + nb22*z_KV;
        const half   * mask_h = (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (Q4I8_D/2);

        int kb0_stop_tile = kb0_stop;
        if (KV_max) {
            kb0_stop_tile = min(kb0_stop_tile, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }

        q4i8_process_tile<ncols1, ncols2, kv_f16, needs_fixup, is_fixup>
            (Q_f2, K_z, V_z, mask_h, dstk, dst_meta, scale, ne01, ne02, stride_Q1, stride_Q2, nb11, nb21, stride_mask,
             jt, kb0_start, kb0_stop_tile);
    };

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        if (kb0_start == 0) {
            tile(std::false_type{}, std::false_type{}); // CUDA block is working on an entire tile.
        } else {
            tile(std::true_type{}, std::false_type{});  // CUDA block is missing the beginning of a tile.
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);

        __syncthreads(); // The next tile overwrites shared memory.
    }

    if (kbc >= kbc_stop) {
        return;
    }

    tile(std::false_type{}, std::true_type{}); // Last index writes its data to fixup buffer to avoid data races with other blocks.
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(BLACKWELL_MMA_AVAILABLE)
}

template <int ncols2, bool kv_f16>
static void q4i8_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int ncols1 = Q4I8_NCOLS/ncols2;
    constexpr int nbytes_shared = q4i8_smem_bytes(ncols1, kv_f16);

    const fattn_kernel_t fattn_kernel = flash_attn_ext_q4i8<ncols1, ncols2, kv_f16>;

    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    const int id = ggml_cuda_get_device();
    if (!shared_memory_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(fattn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared));
        shared_memory_limit_raised[id] = true;
    }

    launch_fattn<Q4I8_D, ncols1, ncols2>
        (ctx, dst, fattn_kernel, Q4I8_NWARPS, nbytes_shared, Q4I8_NBATCH, false, false, true, false, WARP_SIZE);
}

} // namespace

bool ggml_cuda_flash_attn_ext_mma_q4i8_supported(const int device, const ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[device].cc;

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // Only tested on consumer Blackwell (sm_120), the device code is only compiled for that architecture family.
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_BLACKWELL || cc >= GGML_CUDA_CC_RUBIN ||
            ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_BLACKWELL) {
        return false;
    }

    // Large batches only, small batches are latency bound and handled by the other kernels.
    if (Q->ne[1] < 64 || Q->ne[0] != Q4I8_D || sinks || !mask || mask->ne[2] != 1 || (uintptr_t) Q->data % 16 != 0) {
        return false;
    }

    if (K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0) {
        // head size 256, no ALiBi, no softcap, padded KV, 16 byte alignment of K/V/Q/mask rows:
        if (!ggml_cuda_flash_attn_ext_mma_f16_use_q4_0_KV(cc, dst)) {
            return false;
        }
    } else if (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        float max_bias      = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
        if (K->ne[0] != Q4I8_D || V->ne[0] != Q4I8_D || max_bias != 0.0f || logit_softcap != 0.0f || K->ne[1] % FATTN_KQ_STRIDE != 0) {
            return false;
        }
        for (const ggml_tensor * t : {Q, K, V, mask}) {
            if ((uintptr_t) t->data % 16 != 0) {
                return false;
            }
            for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
                if (t->nb[i] % 16 != 0) {
                    return false;
                }
            }
        }
    } else {
        return false;
    }

    // Q columns are processed in groups of ncols2 heads per K/V head without padding.
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    return gqa_ratio % 2 == 0;
}

template <bool kv_f16>
static void ggml_cuda_flash_attn_ext_mma_q4i8_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio % 8 == 0) {
        q4i8_launch<8, kv_f16>(ctx, dst);
    } else if (gqa_ratio % 4 == 0) {
        q4i8_launch<4, kv_f16>(ctx, dst);
    } else {
        GGML_ASSERT(gqa_ratio % 2 == 0);
        q4i8_launch<2, kv_f16>(ctx, dst);
    }
}

void ggml_cuda_flash_attn_ext_mma_q4i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[1]->type == GGML_TYPE_F16) {
        ggml_cuda_flash_attn_ext_mma_q4i8_switch_ncols2<true>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_mma_q4i8_switch_ncols2<false>(ctx, dst);
    }
}

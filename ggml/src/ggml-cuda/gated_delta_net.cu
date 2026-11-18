#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

constexpr int gdn_cols_per_warp = 4;

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Multi-token kernel for S_v == 128 with a scalar gate (no KDA), used for prompt processing (n_tokens >= 8).
// The recurrence is independent per state column. Each CUDA block handles one (head, sequence) and GDN_PF_COLS columns.
// A thread holds GDN_PF_RP rows of GDN_PF_CT columns in registers, GDN_PF_P threads (lanes of one warp)
// share a group of GDN_PF_CT columns and combine their partial dot products with shuffles.
// Rows are distributed in chunks of 4: part p owns rows 4*(p + GDN_PF_P*i) + e, so that the lanes of a column group
// access contiguous memory (coalesced state loads/stores, conflict-free shared memory reads of k and q).
// Holding several columns per thread amortizes the shared memory reads of k and q (the limiter of a 1 column/thread layout).
// k, q, v, g and beta for GDN_PF_T tokens are staged in shared memory with cp.async (double buffered).
// The state is kept as S = sigma * S~ (sigma = product of the gates, renormalized when it leaves [2^-64, 2^64])
// so that the per-token update S~ += k * (delta/sigma) needs a single FMA per element, and the output is computed from
// the old state: o = g*sigma*(S~^T q) + delta*(k.q), which is algebraically identical to (S_new^T q).
#define GDN_PF_S     128
#define GDN_PF_COLS  32                         // columns per block
#define GDN_PF_T     8                          // tokens per staged chunk
#define GDN_PF_P     8                          // threads per column group
#define GDN_PF_CT    2                          // columns per thread
#define GDN_PF_RP    (GDN_PF_S/GDN_PF_P)        // state rows per thread
#define GDN_PF_NT    (GDN_PF_COLS/GDN_PF_CT*GDN_PF_P) // threads per block

static_assert(GDN_PF_RP % 4 == 0 && GDN_PF_CT % 2 == 0 && WARP_SIZE % GDN_PF_P == 0 && GDN_PF_NT % WARP_SIZE == 0, "bad GDN prefill config");

struct gdn_pf_buf {
    float k[GDN_PF_T][GDN_PF_S];
    float q[GDN_PF_T][GDN_PF_S];
    float v[GDN_PF_T][GDN_PF_COLS];
    float g[GDN_PF_T];
    float b[GDN_PF_T];
};

static __device__ __forceinline__ void gdn_pf_cp_async_16(float * dst, const float * src) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"((unsigned int) __cvta_generic_to_shared(dst)), "l"(src));
#else
    *((float4 *) dst) = *((const float4 *) src);
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ void gdn_pf_cp_async_4(float * dst, const float * src) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"((unsigned int) __cvta_generic_to_shared(dst)), "l"(src));
#else
    *dst = *src;
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ void gdn_pf_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

template <int n>
static __device__ __forceinline__ void gdn_pf_cp_async_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}

template <bool keep_rs_t>
__global__ void __launch_bounds__(GDN_PF_NT, 1)
gated_delta_net_prefill_cuda(const float * q,
                             const float * k,
                             const float * v,
                             const float * g,
                             const float * beta,
                             const float * curr_state,
                             float *       dst,
                             float *       state,
                             int64_t       H,
                             int64_t       n_tokens,
                             int64_t       sq1,
                             int64_t       sq2,
                             int64_t       sq3,
                             int64_t       sv1,
                             int64_t       sv2,
                             int64_t       sv3,
                             int64_t       sb1,
                             int64_t       sb2,
                             int64_t       sb3,
                             const uint3   neqk1_magic,
                             const uint3   rq3_magic,
                             float         scale,
                             int64_t       state_slot_stride,
                             int           K) {
    constexpr int S  = GDN_PF_S;
    constexpr int T  = GDN_PF_T;
    constexpr int CT = GDN_PF_CT;
    constexpr int RP = GDN_PF_RP;
    constexpr int NS = T*(2*S/4 + GDN_PF_COLS/4); // 16 byte copies per staged chunk

    __shared__ gdn_pf_buf buf[2];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      tid      = threadIdx.x;
    const int      lane     = tid % WARP_SIZE;
    const int      wid      = tid / WARP_SIZE;
    const int      cl       = (wid*(WARP_SIZE/GDN_PF_P) + lane/GDN_PF_P)*CT; // first column of this thread within block
    const int      p        = lane % GDN_PF_P;                               // row part
    const int      col0     = blockIdx.z*GDN_PF_COLS;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_h = q + iq3*sq3 + iq1*sq1;
    const float * k_h = k + iq3*sq3 + iq1*sq1;
    const float * v_h = v + sequence*sv3 + h_idx*sv1 + col0;
    const int64_t gb_h = sequence*sb3 + h_idx*sb1;

    float * attn_data = dst + (sequence*n_tokens*H + h_idx)*S + col0 + cl;

    // state layout: [col][row] per (head, sequence)
    const int64_t state_offset = (sequence*H + h_idx)*S*S + int64_t(col0 + cl)*S + 4*p;

    const int nchunks = (n_tokens + T - 1) / T;

    auto stage = [&](const int c, gdn_pf_buf & b) {
        const int t0 = c*T;
#pragma unroll
        for (int f0 = 0; f0 < NS; f0 += GDN_PF_NT) {
            const int f  = f0 + tid;
            const int tt = f / (NS/T);
            const int r  = f % (NS/T);
            if ((f0 + GDN_PF_NT > NS && f >= NS) || t0 + tt >= n_tokens) {
                continue;
            }
            const int64_t t = t0 + tt;
            if (r < S/4) {
                gdn_pf_cp_async_16(&b.k[tt][4*r], k_h + t*sq2 + 4*r);
            } else if (r < 2*S/4) {
                const int rq = r - S/4;
                gdn_pf_cp_async_16(&b.q[tt][4*rq], q_h + t*sq2 + 4*rq);
            } else {
                const int rv = r - 2*S/4;
                gdn_pf_cp_async_16(&b.v[tt][4*rv], v_h + t*sv2 + 4*rv);
            }
        }
        if (tid < 2*T) {
            const int tt = tid % T;
            if (t0 + tt < n_tokens) {
                const int64_t off = gb_h + (t0 + tt)*sb2;
                if (tid < T) {
                    gdn_pf_cp_async_4(&b.g[tt], g + off);
                } else {
                    gdn_pf_cp_async_4(&b.b[tt], beta + off);
                }
            }
        }
    };

    ggml_cuda_pdl_sync();

    stage(0, buf[0]);
    gdn_pf_cp_async_commit();

    float s[CT][RP];
#pragma unroll
    for (int j = 0; j < CT; ++j) {
#pragma unroll
        for (int m = 0; m < RP; m += 4) {
            const float4 tmp = *((const float4 *) (curr_state + state_offset + j*S + GDN_PF_P*m));
            s[j][m + 0] = tmp.x;
            s[j][m + 1] = tmp.y;
            s[j][m + 2] = tmp.z;
            s[j][m + 3] = tmp.w;
        }
    }
    float sigma = 1.0f;

    for (int c = 0; c < nchunks; ++c) {
        gdn_pf_buf & b = buf[c % 2];
        if (c + 1 < nchunks) {
            stage(c + 1, buf[(c + 1) % 2]);
        }
        gdn_pf_cp_async_commit();
        gdn_pf_cp_async_wait<1>();
        __syncthreads();

        const int nt = min(T, int(n_tokens - c*T));

        for (int tt = 0; tt < nt; ++tt) {
            const float * kt = b.k[tt] + 4*p;
            const float * qt = b.q[tt] + 4*p;

            // the gate only depends on (head, token): sigma and its renormalization are uniform across the block
            const float g_val    = expf(b.g[tt]);
            const float beta_val = b.b[tt];
            float sigma_new = g_val*sigma;
            const bool renorm = !(sigma_new >= 0x1p-64f && sigma_new <= 0x1p64f);
            sigma_new = renorm ? 1.0f : sigma_new;
            const float inv_sigma_new = 1.0f / sigma_new;

            float kr[RP];
            float kq2[2] = {0.0f, 0.0f};
            float sk[CT][2];
            float sq[CT][2];
#pragma unroll
            for (int j = 0; j < CT; ++j) {
                sk[j][0] = 0.0f; sk[j][1] = 0.0f;
                sq[j][0] = 0.0f; sq[j][1] = 0.0f;
            }
#pragma unroll
            for (int m = 0; m < RP; m += 4) {
                const float4 k4 = *((const float4 *) (kt + GDN_PF_P*m));
                const float4 q4 = *((const float4 *) (qt + GDN_PF_P*m));
                kr[m + 0] = k4.x;
                kr[m + 1] = k4.y;
                kr[m + 2] = k4.z;
                kr[m + 3] = k4.w;
                kq2[0] += k4.x*q4.x;
                kq2[1] += k4.y*q4.y;
                kq2[0] += k4.z*q4.z;
                kq2[1] += k4.w*q4.w;
#pragma unroll
                for (int j = 0; j < CT; ++j) {
                    sk[j][0] += s[j][m + 0]*k4.x;
                    sk[j][1] += s[j][m + 1]*k4.y;
                    sk[j][0] += s[j][m + 2]*k4.z;
                    sk[j][1] += s[j][m + 3]*k4.w;
                    sq[j][0] += s[j][m + 0]*q4.x;
                    sq[j][1] += s[j][m + 1]*q4.y;
                    sq[j][0] += s[j][m + 2]*q4.z;
                    sq[j][1] += s[j][m + 3]*q4.w;
                }
            }
            float rk[CT];
            float rq[CT];
            float kq = kq2[0] + kq2[1];
#pragma unroll
            for (int j = 0; j < CT; ++j) {
                rk[j] = sk[j][0] + sk[j][1];
                rq[j] = sq[j][0] + sq[j][1];
            }
#pragma unroll
            for (int offset = 1; offset < GDN_PF_P; offset *= 2) {
#pragma unroll
                for (int j = 0; j < CT; ++j) {
                    rk[j] += __shfl_xor_sync(0xFFFFFFFF, rk[j], offset, WARP_SIZE);
                }
#pragma unroll
                for (int j = 0; j < CT; ++j) {
                    rq[j] += __shfl_xor_sync(0xFFFFFFFF, rq[j], offset, WARP_SIZE);
                }
                kq += __shfl_xor_sync(0xFFFFFFFF, kq, offset, WARP_SIZE);
            }

            float vv[CT];
#pragma unroll
            for (int j = 0; j < CT; j += 2) {
                const float2 tmp = *((const float2 *) &b.v[tt][cl + j]);
                vv[j + 0] = tmp.x;
                vv[j + 1] = tmp.y;
            }

            float delta[CT];
            float attn[CT];
#pragma unroll
            for (int j = 0; j < CT; ++j) {
                const float kv = sigma*rk[j];
                delta[j] = (vv[j] - g_val*kv)*beta_val;
                attn[j]  = g_val*sigma*rq[j] + delta[j]*kq;
            }

            if (renorm) {
                const float f = g_val*sigma;
#pragma unroll
                for (int j = 0; j < CT; ++j) {
#pragma unroll
                    for (int m = 0; m < RP; ++m) {
                        s[j][m] *= f;
                    }
                }
            }
            sigma = sigma_new;
#pragma unroll
            for (int j = 0; j < CT; ++j) {
                const float d = delta[j]*inv_sigma_new;
#pragma unroll
                for (int m = 0; m < RP; ++m) {
                    s[j][m] += kr[m]*d;
                }
            }

            const int t = c*T + tt;
            if (p == 0) {
#pragma unroll
                for (int j = 0; j < CT; j += 2) {
                    *((float2 *) &attn_data[int64_t(t)*S*H + j]) = make_float2(attn[j + 0]*scale, attn[j + 1]*scale);
                }
            }

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * st = state + target_slot*state_slot_stride + state_offset;
#pragma unroll
                    for (int j = 0; j < CT; ++j) {
#pragma unroll
                        for (int m = 0; m < RP; m += 4) {
                            *((float4 *) (st + j*S + GDN_PF_P*m)) = make_float4(
                                sigma*s[j][m + 0], sigma*s[j][m + 1], sigma*s[j][m + 2], sigma*s[j][m + 3]);
                        }
                    }
                }
            }
        }
        __syncthreads(); // buffer is reused by the stage for chunk c + 2
    }

    if constexpr (!keep_rs_t) {
        float * st = state + state_offset;
#pragma unroll
        for (int j = 0; j < CT; ++j) {
#pragma unroll
            for (int m = 0; m < RP; m += 4) {
                *((float4 *) (st + j*S + GDN_PF_P*m)) = make_float4(sigma*s[j][m + 0], sigma*s[j][m + 1], sigma*s[j][m + 2], sigma*s[j][m + 3]);
            }
        }
    }
}

// Whether the multi-token kernel can be used: head size 128, scalar gate, >= 8 tokens, 16 byte aligned q/k/v rows.
// It is faster than the per-token kernel from 2 tokens on; the small batches of speculative decoding (<= 4 tokens)
// deliberately keep using the per-token kernel so that the decoding numerics are unchanged.
static bool gdn_prefill_supported(const ggml_tensor * src_q, const ggml_tensor * src_k, const ggml_tensor * src_v,
        const int64_t S_v, const bool kda, const int64_t n_tokens) {
    return S_v == GDN_PF_S && !kda && n_tokens >= 8 &&
        ggml_cuda_is_aligned(src_q, 16) && ggml_cuda_is_aligned(src_k, 16) && ggml_cuda_is_aligned(src_v, 16);
}

template <bool keep_rs_t>
static void launch_gated_delta_net_prefill(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const dim3 grid_dims(H, n_seqs, GDN_PF_S/GDN_PF_COLS);
    const dim3 block_dims(GDN_PF_NT, 1, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_prefill_cuda<keep_rs_t>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
}


template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    if (!kda && S_v == 128 && n_tokens > 8 && ggml_cuda_op_gated_delta_net_seq(ctx, dst, nullptr, cache)) {
        return;
    }

    if (!kda && S_v == 128 && n_tokens <= 8) {
        ggml_cuda_gdn_short_args a = {};
        a.q = q_d; a.k = k_d;
        a.sq1 = sq1; a.sq2 = sq2; a.sq3 = sq3;
        a.qk_ne1 = neqk1; a.qk_ne3 = neq3;
        a.l2 = false;
        a.state = s_d; a.state_rows = nullptr; a.state_row_stride = 0;
        if (ggml_cuda_op_gated_delta_net_short(ctx, dst, a, cache)) {
            return;
        }
    }


    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    const bool use_prefill = gdn_prefill_supported(src_q, src_k, src_v, S_v, kda, n_tokens) &&
        reinterpret_cast<uintptr_t>(s_d) % 16 == 0 && reinterpret_cast<uintptr_t>(state_d) % 16 == 0 &&
        (!keep_rs || state_slot_stride % 4 == 0);

    if (use_prefill) {
        if (keep_rs) {
            launch_gated_delta_net_prefill<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net_prefill<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}


// Short-sequence (verify/decode) variant for S_v = 128, scalar gate: all per-token inputs are loaded up front so
// the sequential recurrence only waits on the warp reductions. Optionally
//   - L2-normalizes q and k in-kernel (replaces rms_norm(eps/n) * 1/sqrt(n) nodes), and
//   - reads the initial state straight from the cache row given by a device index (replaces the get_rows copy).
// Each warp owns one state column for its head and reads it completely before writing it, so the state source may
// alias the snapshot destination (single sequence).
template <int max_t, bool keep_rs_t>
__global__ void __launch_bounds__(128, 2)
gated_delta_net_short_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
                           const float * curr_state, const int32_t * state_rows, const int64_t state_row_stride,
                           float * dst, float * state, const int64_t H, const int n_tokens,
                           const int64_t sq1, const int64_t sq2, const int64_t sq3,
                           const int64_t sv1, const int64_t sv2, const int64_t sv3,
                           const int64_t sb1, const int64_t sb2, const int64_t sb3,
                           const uint3 neqk1_magic, const uint3 rq3_magic, const float scale,
                           const int l2, const float eps_q, const float scale_q, const float eps_k, const float scale_k,
                           const int64_t state_slot_stride, const int K) {
    constexpr int S_v = 128;
    constexpr int R   = S_v / WARP_SIZE; // rows per lane

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;
    state += (sequence * H + h_idx) * S_v * S_v;

    ggml_cuda_pdl_lc();

    // The state (also a gathered cache row) is only read after the wait: an earlier node of the same graph may still
    // be writing it (e.g. the scale_inplace that zeroes the row of a new sequence).
    ggml_cuda_pdl_sync();

    float s_shard[R];
    {
        const float * cs = state_rows ? curr_state + (int64_t) state_rows[sequence] * state_row_stride + h_idx * S_v * S_v + col * S_v
                                      : curr_state + sequence * H * S_v * S_v + h_idx * S_v * S_v + col * S_v;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s_shard[r] = cs[r * WARP_SIZE + lane];
        }
    }

    float q_reg[max_t][R];
    float k_reg[max_t][R];
    float v_val[max_t];
    float g_val[max_t];
    float b_val[max_t];

#pragma unroll
    for (int t = 0; t < max_t; ++t) {
        if (t < n_tokens) {
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                q_reg[t][r] = q_t[r * WARP_SIZE + lane];
                k_reg[t][r] = k_t[r * WARP_SIZE + lane];
            }
            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            v_val[t] = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col];
            g_val[t] = expf(g[gb_offset]);
            b_val[t] = beta[gb_offset];
        }
    }

    if (l2) {
#pragma unroll
        for (int t = 0; t < max_t; ++t) {
            if (t < n_tokens) {
                float sq = 0.0f;
                float sk = 0.0f;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    sq += q_reg[t][r] * q_reg[t][r];
                    sk += k_reg[t][r] * k_reg[t][r];
                }
                sq = warp_reduce_sum<WARP_SIZE>(sq);
                sk = warp_reduce_sum<WARP_SIZE>(sk);
                const float fq = rsqrtf(sq / S_v + eps_q);
                const float fk = rsqrtf(sk / S_v + eps_k);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    q_reg[t][r] = scale_q * (fq * q_reg[t][r]);
                    k_reg[t][r] = scale_k * (fk * k_reg[t][r]);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < max_t; ++t) {
        if (t < n_tokens) {
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                kv_shard += s_shard[r] * k_reg[t][r];
            }
            const float kv_col    = warp_reduce_sum<WARP_SIZE>(kv_shard);
            const float delta_col = (v_val[t] - g_val[t] * kv_col) * b_val[t];

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                s_shard[r]    = g_val[t] * s_shard[r] + k_reg[t][r] * delta_col;
                attn_partial += s_shard[r] * q_reg[t][r];
            }
            const float attn_col = warp_reduce_sum<WARP_SIZE>(attn_partial);
            if (lane == 0) {
                attn_data[t * S_v * H + col] = attn_col * scale;
            }

            if constexpr (keep_rs_t) {
                const int target_slot = n_tokens - 1 - t;
                if (target_slot < K) {
                    float * st = state + target_slot * state_slot_stride + col * S_v;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        st[r * WARP_SIZE + lane] = s_shard[r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            state[col * S_v + r * WARP_SIZE + lane] = s_shard[r];
        }
    }
}

bool ggml_cuda_op_gated_delta_net_short(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gdn_short_args & a, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    const ggml_tensor * src_v    = dst->src[2];
    const ggml_tensor * src_g    = dst->src[3];
    const ggml_tensor * src_beta = dst->src[4];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    if (S_v != 128 || n_tokens > 8 || src_g->ne[0] != 1 || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != WARP_SIZE ||
        !ggml_is_contiguous(src_g) || !ggml_is_contiguous(src_beta) || !ggml_is_contiguous_rows(src_v) ||
        a.qk_ne1 <= 0 || H % a.qk_ne1 != 0 || n_seqs % a.qk_ne3 != 0 || (a.state_rows && n_seqs != 1)) {
        return false;
    }

    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    float * dst_d             = (float *) dst->data;
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const int64_t sv1 = src_v->nb[1] / sizeof(float);
    const int64_t sv2 = src_v->nb[2] / sizeof(float);
    const int64_t sv3 = src_v->nb[3] / sizeof(float);
    const int64_t sb1 = src_beta->nb[1] / sizeof(float);
    const int64_t sb2 = src_beta->nb[2] / sizeof(float);
    const int64_t sb3 = src_beta->nb[3] / sizeof(float);

    const uint3 neqk1_magic = init_fastdiv_values(a.qk_ne1);
    const uint3 rq3_magic   = init_fastdiv_values(n_seqs / a.qk_ne3);
    const float scale       = 1.0f / sqrtf((float) S_v);

    const int num_warps = 4;
    const dim3 grid_dims(H, n_seqs, S_v / num_warps);
    const dim3 block_dims(WARP_SIZE, num_warps, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto launch = [&](auto MT, auto KEEP) {
        constexpr int  kMT   = decltype(MT)::value;
        constexpr bool kKeep = decltype(KEEP)::value;
        ggml_cuda_kernel_launch(gated_delta_net_short_cuda<kMT, kKeep>, launch_params,
            a.q, a.k, (const float *) src_v->data, (const float *) src_g->data, (const float *) src_beta->data,
            a.state, a.state_rows, a.state_row_stride, dst_d, state_d, H, (int) n_tokens,
            a.sq1, a.sq2, a.sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale,
            a.l2 ? 1 : 0, a.eps_q, a.scale_q, a.eps_k, a.scale_k, state_slot_stride, K);
    };
    auto launch_keep = [&](auto MT) {
        if (keep_rs) {
            launch(MT, std::true_type{});
        } else {
            launch(MT, std::false_type{});
        }
    };
    if (n_tokens <= 1) {
        launch_keep(std::integral_constant<int, 1>{});
    } else if (n_tokens <= 4) {
        launch_keep(std::integral_constant<int, 4>{});
    } else {
        launch_keep(std::integral_constant<int, 8>{});
    }
    return true;
}

// Gated-delta-net gate inputs from one launch (replaces 2 BF16 mat-vecs + add + softplus + mul + sigmoid):
//   g[t][r]    = softplus(dot(Wa[r], x[t]) + dt[r]) * a[r]
//   beta[t][r] = sigmoid(dot(Wb[r], x[t]))
// One block per row pair (Wa[r], Wb[r]) so each x load feeds both dot products.
template <int max_t>
static __global__ void __launch_bounds__(256, 1)
gdn_gates_bf16_cuda(const nv_bfloat16 * wa, const nv_bfloat16 * wb, const int64_t wa_s1, const int64_t wb_s1,
                    const float * x, const int64_t x_s1, const int ncols, const int n_t,
                    const float * dt, const float * a, float * g_out, const int64_t g_s1, float * b_out, const int64_t b_s1) {
    constexpr int nthreads = 256;
    const int r   = blockIdx.x;
    const int tid = threadIdx.x;

    const nv_bfloat16 * war = wa + r * wa_s1;
    const nv_bfloat16 * wbr = wb + r * wb_s1;

    float acc_a[max_t] = { 0.0f };
    float acc_b[max_t] = { 0.0f };

    ggml_cuda_pdl_lc();

    // weights are constant: fetch them before waiting on the producer of x
    constexpr int max_iter = 8;
    uint2 pa_r[max_iter];
    uint2 pb_r[max_iter];
#pragma unroll
    for (int it = 0; it < max_iter; ++it) {
        const int c = 4 * tid + it * 4 * nthreads;
        if (c < ncols) {
            pa_r[it] = *(const uint2 *) (war + c);
            pb_r[it] = *(const uint2 *) (wbr + c);
        }
    }

    ggml_cuda_pdl_sync();

#pragma unroll
    for (int it = 0; it < max_iter; ++it) {
        const int c = 4 * tid + it * 4 * nthreads;
        if (c >= ncols) {
            break;
        }
        const uint2 pa = pa_r[it];
        const uint2 pb = pb_r[it];
        const float2 a01 = __bfloat1622float2(*(const nv_bfloat162 *) &pa.x);
        const float2 a23 = __bfloat1622float2(*(const nv_bfloat162 *) &pa.y);
        const float2 b01 = __bfloat1622float2(*(const nv_bfloat162 *) &pb.x);
        const float2 b23 = __bfloat1622float2(*(const nv_bfloat162 *) &pb.y);
#pragma unroll
        for (int t = 0; t < max_t; ++t) {
            if (t < n_t) {
                const float4 xv = *(const float4 *) (x + t * x_s1 + c);
                acc_a[t] += a01.x * xv.x + a01.y * xv.y + a23.x * xv.z + a23.y * xv.w;
                acc_b[t] += b01.x * xv.x + b01.y * xv.y + b23.x * xv.z + b23.y * xv.w;
            }
        }
    }

    __shared__ float s_red[2 * max_t][nthreads / WARP_SIZE];
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
#pragma unroll
    for (int t = 0; t < max_t; ++t) {
        acc_a[t] = warp_reduce_sum<WARP_SIZE>(acc_a[t]);
        acc_b[t] = warp_reduce_sum<WARP_SIZE>(acc_b[t]);
        if (lane == 0) {
            s_red[2 * t + 0][warp] = acc_a[t];
            s_red[2 * t + 1][warp] = acc_b[t];
        }
    }
    __syncthreads();
    if (tid < 2 * max_t && tid / 2 < n_t) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < nthreads / WARP_SIZE; ++w) {
            sum += s_red[tid][w];
        }
        const int t = tid / 2;
        if (tid % 2 == 0) {
            const float z  = sum + dt[r];
            const float sp = z > 20.0f ? z : logf(1.0f + expf(z));
            g_out[t * g_s1 + r] = sp * a[r];
        } else {
            b_out[t * b_s1 + r] = 1.0f / (1.0f + expf(-sum));
        }
    }
}

bool ggml_cuda_op_gdn_gates(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_gates_args & p) {
    if (p.ncols % 4 != 0 || p.ncols > 8 * 4 * 256 || p.n_t < 1 || p.n_t > 8 || p.wa_s1 % 4 != 0 || p.wb_s1 % 4 != 0 || p.x_s1 % 4 != 0 ||
        ((uintptr_t) p.wa | (uintptr_t) p.wb) % 8 != 0 || (uintptr_t) p.x % 16 != 0) {
        return false;
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(dim3(p.nrows), dim3(256), 0, ctx.stream());
    auto launch = [&](auto MT) {
        constexpr int kMT = decltype(MT)::value;
        ggml_cuda_kernel_launch(gdn_gates_bf16_cuda<kMT>, launch_params,
            (const nv_bfloat16 *) p.wa, (const nv_bfloat16 *) p.wb, p.wa_s1, p.wb_s1, p.x, p.x_s1, p.ncols, p.n_t,
            p.dt, p.a, p.g, p.g_s1, p.b, p.b_s1);
    };
    if (p.n_t == 1) {
        launch(std::integral_constant<int, 1>{});
    } else if (p.n_t <= 4) {
        launch(std::integral_constant<int, 4>{});
    } else {
        launch(std::integral_constant<int, 8>{});
    }
    return true;
}

// Long-sequence (prefill) variant for S_v = 128, scalar gate. Each state column is owned by a group of P lanes that
// hold 128/P of its rows in registers, so the per-token dot products are mostly thread-local FMA chains and need only
// log2(P) shuffle steps (the one-column-per-warp kernel spends its time in 5-step warp reductions). The per-token q/k
// rows (shared by all columns of the head), v, exp(g) and beta are staged through double-buffered shared memory in
// chunks of T tokens, prefetched into registers one chunk ahead. Optionally q/k are L2-normalized (per-token factors
// computed once per chunk and folded into the dot products) and the initial state is gathered from a cache row.
// replay mode of the long kernel (GGML_OP_GATED_DELTA_NET_REPLAY): the initial state is the post-replay state in
// `x` when rep[s] > 0 (written by the replay-only launch of the replay kernel). With chain, the loaded state is
// state_out when folding (the no-fold copy is done by the replay-only launch) and the ring rows of all tokens are
// written; otherwise state_out is the final state.
struct gdn_seq_replay_params {
    const int32_t * rep;   // [count, fold] per sequence
    const float   * x;     // post-replay state (when count > 0)
    float * out;      const int32_t * out_rows;      int64_t out_row_stride;
    float * ring_out; const int32_t * ring_out_rows;
    int chain;             // n_tokens <= W: write the ring rows of all tokens, state_out = x/input (fold) or input
    int H_k;
};

template <int P, int W, int T, bool keep_rs_t, bool replay_t = false>
__global__ void __launch_bounds__(W * WARP_SIZE, 1)
gated_delta_net_seq_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
                         const float * curr_state, const int32_t * state_rows, const int64_t state_row_stride,
                         float * dst, float * state, const int64_t H, const int n_tokens,
                         const int64_t sq1, const int64_t sq2, const int64_t sq3,
                         const int64_t sv1, const int64_t sv2, const int64_t sv3,
                         const int64_t sb1, const int64_t sb2, const int64_t sb3,
                         const uint3 neqk1_magic, const uint3 rq3_magic, const float scale,
                         const int l2, const float eps_q, const float scale_q, const float eps_k, const float scale_k,
                         const int64_t state_slot_stride, const int K, const gdn_seq_replay_params rp) {
    constexpr int S    = 128;
    constexpr int RL   = S / P;                 // rows per lane
    constexpr int RP   = RL + 4;                // padded row chunk: the P chunks of a row start in different banks
    constexpr int NW   = W;
    constexpr int NT   = W * WARP_SIZE;
    constexpr int NCOL = NT / P;                // columns per block
    constexpr int LQ   = T * S / NT;            // q (and k) values per thread per chunk
    constexpr int LV   = (T * NCOL + NT - 1) / NT;
    static_assert(S % P == 0 && RL % 4 == 0 && T * S % NT == 0 && 2 * T <= NT, "bad tiling");

    const int h_idx    = blockIdx.x;
    const int sequence = blockIdx.y;
    const int lane     = threadIdx.x;
    const int warp     = threadIdx.y;
    const int tid      = warp * WARP_SIZE + lane;
    const int lcol     = tid / P;               // column within the block
    const int part     = tid % P;               // row chunk of that column
    const int col      = blockIdx.z * NCOL + lcol;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    __shared__ float4 qs[2][T][P * RP / 4];
    __shared__ float4 ks[2][T][P * RP / 4];
    __shared__ float  vs[2][T][NCOL];
    __shared__ float  gs[2][T];
    __shared__ float  bs[2][T];
    __shared__ float  fqs[2][T];
    __shared__ float  fks[2][T];

    const float * qb = q + iq3 * sq3 + iq1 * sq1;
    const float * kb = k + iq3 * sq3 + iq1 * sq1;
    const float * vb = v + sequence * sv3 + h_idx * sv1 + blockIdx.z * NCOL;
    const int64_t gb = sequence * sb3 + h_idx * sb1;

    float * attn_data = dst + ((int64_t) sequence * n_tokens * H + h_idx) * S;
    if constexpr (replay_t) {
        state = (rp.out_rows ? rp.out + (int64_t) rp.out_rows[sequence] * rp.out_row_stride
                             : rp.out + (int64_t) sequence * H * S * S) + (int64_t) h_idx * S * S;
    } else {
        state += ((int64_t) sequence * H + h_idx) * S * S;
    }

    float pq[LQ], pk[LQ], pv[LV], pg = 0.0f;

    auto prefetch = [&](const int t0) {
#pragma unroll
        for (int i = 0; i < LQ; ++i) {
            const int e = tid + i * NT;
            const int t = t0 + e / S;
            pq[i] = t < n_tokens ? qb[t * sq2 + e % S] : 0.0f;
            pk[i] = t < n_tokens ? kb[t * sq2 + e % S] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < LV; ++i) {
            const int e = tid + i * NT;
            const int t = t0 + e / NCOL;
            pv[i] = e < T * NCOL && t < n_tokens ? vb[t * sv2 + e % NCOL] : 0.0f;
        }
        if (tid < 2 * T) {
            const int t = t0 + tid % T;
            pg = t < n_tokens ? (tid < T ? g[gb + t * sb2] : beta[gb + t * sb2]) : 0.0f;
        }
    };
    // stage the prefetched chunk into smem buffer `buf`, then (l2) compute its per-token q/k normalization factors
    auto stage = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < LQ; ++i) {
            const int e = tid + i * NT;
            const int r = e % S;
            ((float *) qs[buf][e / S])[(r / RL) * RP + r % RL] = pq[i];
            ((float *) ks[buf][e / S])[(r / RL) * RP + r % RL] = pk[i];
        }
#pragma unroll
        for (int i = 0; i < LV; ++i) {
            const int e = tid + i * NT;
            if (e < T * NCOL) {
                vs[buf][e / NCOL][e % NCOL] = pv[i];
            }
        }
        if (tid < T) {
            gs[buf][tid] = expf(pg);
        } else if (tid < 2 * T) {
            bs[buf][tid - T] = pg;
        }
        if (l2) {
            __syncthreads();
            for (int row = warp; row < 2 * T; row += NW) {
                const float * x = (const float *) (row < T ? qs[buf][row] : ks[buf][row - T]);
                float ss = 0.0f;
#pragma unroll
                for (int j = 0; j < S / WARP_SIZE; ++j) {
                    const int r = j * WARP_SIZE + lane;
                    const float xv = x[(r / RL) * RP + r % RL];
                    ss += xv * xv;
                }
                ss = warp_reduce_sum<WARP_SIZE>(ss);
                if (lane == 0) {
                    if (row < T) {
                        fqs[buf][row] = scale_q * rsqrtf(ss / S + eps_q);
                    } else {
                        fks[buf][row - T] = scale_k * rsqrtf(ss / S + eps_k);
                    }
                }
            }
        }
    };

    // state is stored transposed: M[col][i] = S[i][col]; this thread holds rows [part*RL, part*RL + RL) of its column
    float st[RL];
    auto load_state = [&](const float * base) {
        const float4 * cs = (const float4 *) (base + h_idx * S * S + (int64_t) col * S + part * RL);
#pragma unroll
        for (int j = 0; j < RL / 4; ++j) {
            const float4 x = cs[j];
            st[4 * j + 0] = x.x; st[4 * j + 1] = x.y; st[4 * j + 2] = x.z; st[4 * j + 3] = x.w;
        }
    };
    ggml_cuda_pdl_sync();

    if constexpr (replay_t) {
        const float * src = state_rows ? curr_state + (int64_t) state_rows[sequence] * state_row_stride
                                       : curr_state + (int64_t) sequence * H * S * S;
        const int p = rp.rep[2 * sequence + 0];
        load_state(p > 0 ? rp.x + (int64_t) sequence * H * S * S : src);
        // fold: the loaded (post-replay) state is state_out, not rewritten when it is the unchanged source row
        if (rp.chain && rp.rep[2 * sequence + 1] != 0 && !(p == 0 && state == src + (int64_t) h_idx * S * S)) {
            float4 * so = (float4 *) (state + (int64_t) col * S + part * RL);
#pragma unroll
            for (int j = 0; j < RL / 4; ++j) {
                so[j] = make_float4(st[4 * j + 0], st[4 * j + 1], st[4 * j + 2], st[4 * j + 3]);
            }
        }
    } else {
    load_state(state_rows ? curr_state + (int64_t) state_rows[sequence] * state_row_stride
                          : curr_state + (int64_t) sequence * H * S * S);
    }

    prefetch(0);
    stage(0);
    __syncthreads();

    const int n_chunks = (n_tokens + T - 1) / T;
    for (int ch = 0; ch < n_chunks; ++ch) {
        const int buf = ch & 1;
        if (ch + 1 < n_chunks) {
            prefetch((ch + 1) * T);
        }
        const int t_end = min(T, n_tokens - ch * T);
        for (int tt = 0; tt < t_end; ++tt) {
            const int     t  = ch * T + tt;
            const float   gv = gs[buf][tt];
            const float   bv = bs[buf][tt];
            const float   fk = l2 ? fks[buf][tt] : 1.0f;
            const float   fq = l2 ? fqs[buf][tt] : 1.0f;
            const float4 * k4 = ks[buf][tt] + part * (RP / 4);
            const float4 * q4 = qs[buf][tt] + part * (RP / 4);

            float kv0 = 0.0f, kv1 = 0.0f, kv2 = 0.0f, kv3 = 0.0f;
#pragma unroll
            for (int j = 0; j < RL / 4; ++j) {
                const float4 kk = k4[j];
                kv0 += st[4 * j + 0] * kk.x;
                kv1 += st[4 * j + 1] * kk.y;
                kv2 += st[4 * j + 2] * kk.z;
                kv3 += st[4 * j + 3] * kk.w;
            }
            float kv = (kv0 + kv1) + (kv2 + kv3);
#pragma unroll
            for (int o = P / 2; o > 0; o >>= 1) {
                kv += __shfl_xor_sync(0xffffffff, kv, o, WARP_SIZE);
            }
            // with normalized k' = fk*k: kv' = fk*kv and S' = g*S + k'*delta = g*S + k*(fk*delta)
            const float delta = (vs[buf][tt][lcol] - gv * (fk * kv)) * bv * fk;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (int j = 0; j < RL / 4; ++j) {
                const float4 kk = k4[j];
                const float4 qq = q4[j];
                st[4 * j + 0] = gv * st[4 * j + 0] + kk.x * delta;
                st[4 * j + 1] = gv * st[4 * j + 1] + kk.y * delta;
                st[4 * j + 2] = gv * st[4 * j + 2] + kk.z * delta;
                st[4 * j + 3] = gv * st[4 * j + 3] + kk.w * delta;
                a0 += st[4 * j + 0] * qq.x;
                a1 += st[4 * j + 1] * qq.y;
                a2 += st[4 * j + 2] * qq.z;
                a3 += st[4 * j + 3] * qq.w;
            }
            float ap = (a0 + a1) + (a2 + a3);
#pragma unroll
            for (int o = P / 2; o > 0; o >>= 1) {
                ap += __shfl_xor_sync(0xffffffff, ap, o, WARP_SIZE);
            }
            if (part == 0) {
                attn_data[(int64_t) t * S * H + col] = fq * ap * scale;
            }
            if constexpr (keep_rs_t) {
                const int target_slot = n_tokens - 1 - t;
                if (target_slot < K) {
                    float4 * so = (float4 *) (state + target_slot * state_slot_stride + (int64_t) col * S + part * RL);
#pragma unroll
                    for (int j = 0; j < RL / 4; ++j) {
                        so[j] = make_float4(st[4 * j + 0], st[4 * j + 1], st[4 * j + 2], st[4 * j + 3]);
                    }
                }
            }
        }
        if (ch + 1 < n_chunks) {
            stage(buf ^ 1);
            __syncthreads();
        }
    }

    if constexpr (!keep_rs_t) {
        if (!replay_t || !rp.chain) {
            float4 * so = (float4 *) (state + (int64_t) col * S + part * RL);
#pragma unroll
            for (int j = 0; j < RL / 4; ++j) {
                so[j] = make_float4(st[4 * j + 0], st[4 * j + 1], st[4 * j + 2], st[4 * j + 3]);
            }
        }
    }

    if constexpr (replay_t) {
        // ring rows of all tokens (chain): k [S, H_k] (normalized) | v [S, H] | g [H] | beta [H]
        const int64_t F     = (int64_t) S * rp.H_k + (int64_t) S * H + 2 * H;
        const int64_t off_v = (int64_t) S * rp.H_k;
        const int64_t off_g = off_v + (int64_t) S * H;
        for (int t = 0; t < (rp.chain ? n_tokens : 0); ++t) {
            float * row = rp.ring_out + (rp.ring_out_rows ? (int64_t) rp.ring_out_rows[sequence * n_tokens + t]
                                                          : (int64_t) sequence * n_tokens + t) * F;
            if (part == 0) {
                row[off_v + (int64_t) h_idx * S + col] = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col];
            }
            if (blockIdx.z == 0 && tid == 0) {
                row[off_g + h_idx]     = g[gb + t * sb2];
                row[off_g + H + h_idx] = beta[gb + t * sb2];
            }
            if (h_idx < rp.H_k && blockIdx.z == 0 && warp == 0) {
                const float * kt = kb + t * sq2;
                float kr[S / WARP_SIZE];
                float ss = 0.0f;
#pragma unroll
                for (int j = 0; j < S / WARP_SIZE; ++j) {
                    kr[j] = kt[j * WARP_SIZE + lane];
                    ss += kr[j] * kr[j];
                }
                float f = 1.0f;
                if (l2) {
                    ss = warp_reduce_sum<WARP_SIZE>(ss);
                    f  = scale_k * rsqrtf(ss / S + eps_k);
                }
#pragma unroll
                for (int j = 0; j < S / WARP_SIZE; ++j) {
                    row[(int64_t) h_idx * S + j * WARP_SIZE + lane] = f * kr[j];
                }
            }
        }
    }
}

bool ggml_cuda_op_gated_delta_net_seq(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                      const ggml_cuda_gdn_short_args * args, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    const ggml_tensor * src_q    = dst->src[0];
    const ggml_tensor * src_k    = dst->src[1];
    const ggml_tensor * src_v    = dst->src[2];
    const ggml_tensor * src_g    = dst->src[3];
    const ggml_tensor * src_beta = dst->src[4];
    const ggml_tensor * src_s    = dst->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    ggml_cuda_gdn_short_args a = {};
    if (args) {
        a = *args;
    } else {
        a.q = (const float *) src_q->data; a.k = (const float *) src_k->data;
        a.sq1 = src_q->nb[1] / sizeof(float); a.sq2 = src_q->nb[2] / sizeof(float); a.sq3 = src_q->nb[3] / sizeof(float);
        a.qk_ne1 = src_q->ne[1]; a.qk_ne3 = src_q->ne[3];
        a.l2 = false;
        a.state = (const float *) src_s->data; a.state_rows = nullptr; a.state_row_stride = 0;
        if (!ggml_is_contiguous_rows(src_q) || !ggml_is_contiguous_rows(src_k) || !ggml_are_same_stride(src_q, src_k) ||
            !ggml_is_contiguous(src_s)) {
            return false;
        }
    }

    if (S_v != 128 || n_tokens <= 8 || n_tokens > INT32_MAX || src_g->ne[0] != 1 ||
        ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != WARP_SIZE ||
        !ggml_is_contiguous(src_g) || !ggml_is_contiguous(src_beta) || !ggml_is_contiguous_rows(src_v) ||
        a.qk_ne1 <= 0 || H % a.qk_ne1 != 0 || n_seqs % a.qk_ne3 != 0 || (a.state_rows && n_seqs != 1)) {
        return false;
    }

    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    float * dst_d             = (float *) dst->data;
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    constexpr int P = 2, W = 4, T = 8;
    const dim3 grid_dims(H, n_seqs, S_v / (W * WARP_SIZE / P));
    const dim3 block_dims(WARP_SIZE, W, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());
    const uint3 neqk1_magic = init_fastdiv_values(a.qk_ne1);
    const uint3 rq3_magic   = init_fastdiv_values(n_seqs / a.qk_ne3);
    const float scale       = 1.0f / sqrtf((float) S_v);

    auto launch = [&](auto KEEP) {
        ggml_cuda_kernel_launch(gated_delta_net_seq_cuda<P, W, T, decltype(KEEP)::value>, lp,
            a.q, a.k, (const float *) src_v->data, (const float *) src_g->data, (const float *) src_beta->data,
            a.state, a.state_rows, a.state_row_stride, dst_d, state_d, H, (int) n_tokens,
            a.sq1, a.sq2, a.sq3,
            (int64_t) (src_v->nb[1] / sizeof(float)), (int64_t) (src_v->nb[2] / sizeof(float)), (int64_t) (src_v->nb[3] / sizeof(float)),
            (int64_t) (src_beta->nb[1] / sizeof(float)), (int64_t) (src_beta->nb[2] / sizeof(float)), (int64_t) (src_beta->nb[3] / sizeof(float)),
            neqk1_magic, rq3_magic, scale, a.l2 ? 1 : 0, a.eps_q, a.scale_q, a.eps_k, a.scale_k, state_slot_stride, K,
            gdn_seq_replay_params{});
    };
    if (keep_rs) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
#ifndef GDN_REPLAY_COLS
#define GDN_REPLAY_COLS 4 // state columns per warp in the replay kernel
#endif

// GGML_OP_GATED_DELTA_NET_REPLAY: rollback of the recurrent state by replay instead of per-token snapshots.
// The cache keeps one state per sequence (a "base") plus the inputs (ring rows) of the tokens processed after it; a
// call replays the pending rows from the base, then processes its new tokens. A rollback only drops pending rows.
// The per-token arithmetic is exactly that of gated_delta_net_short_cuda (the replayed steps run the same code without
// the output), so the replayed state is bitwise the state the snapshot scheme would have kept.
// Each warp owns C state columns of one head and reads them completely before writing them, so the state source may
// alias the state destination of the same sequence.

// warp_reduce_sum (xor butterfly 16, 8, 4, 2, 1; every lane ends with the same bits) of C values per lane at once, with
// the same pairing at every stage: the first stages merge two columns into one register (lanes whose partner holds the
// other column), so C = 4 takes 6 shuffles instead of 20. Afterwards column c is complete in lane gdn_rs_lane<C>(c).
template <int C>
static __device__ __forceinline__ int gdn_rs_lane(const int c) {
    return C == 4 ? (c & 1) * 16 + (c >> 1) * 8 : C == 2 ? c * 16 : 0;
}

template <int C>
static __device__ __forceinline__ float gdn_reduce_packed(const float (&x)[C], const int lane) {
    static_assert(C == 1 || C == 2 || C == 4, "bad column count");
    if constexpr (C == 1) {
        return warp_reduce_sum<WARP_SIZE>(x[0]);
    } else {
        // stage 16: lanes < 16 finish column 0 (2), lanes >= 16 column 1 (3)
        const bool hi = lane & 16;
        float a = (hi ? x[1] : x[0]) + __shfl_xor_sync(0xffffffff, hi ? x[0] : x[1], 16, WARP_SIZE);
        if constexpr (C == 4) {
            float b = (hi ? x[3] : x[2]) + __shfl_xor_sync(0xffffffff, hi ? x[2] : x[3], 16, WARP_SIZE);
            // stage 8: lanes with bit 3 clear keep a (columns 0/1), the others b (columns 2/3)
            const bool b8 = lane & 8;
            a = (b8 ? b : a) + __shfl_xor_sync(0xffffffff, b8 ? a : b, 8, WARP_SIZE);
        } else {
            a += __shfl_xor_sync(0xffffffff, a, 8, WARP_SIZE);
        }
        a += __shfl_xor_sync(0xffffffff, a, 4, WARP_SIZE);
        a += __shfl_xor_sync(0xffffffff, a, 2, WARP_SIZE);
        a += __shfl_xor_sync(0xffffffff, a, 1, WARP_SIZE);
        return a;
    }
}

// Each warp owns C state columns of one head (C independent recurrences per warp); the per-column arithmetic is that of
// gated_delta_net_short_cuda (lane l holds rows 32*r + l, sequential partial sums over r, then the same warp reduction).
// The q/k rows (the same for all columns of a head) are staged once per block in shared memory, and the l2 norms are
// computed once per token by one warp (same arithmetic as in the short kernel).
template <int max_t, int C>
__global__ void __launch_bounds__(128, 6)
gated_delta_net_replay_cuda(const ggml_cuda_gdn_replay_args a, const int64_t H, const int n_tokens, const int n_len,
                            const uint3 neqk1_magic, const float scale, const bool replay_only) {
    constexpr int S_v = 128;
    constexpr int R   = S_v / WARP_SIZE; // rows per lane
    constexpr int NW  = 4;               // warps per block
    constexpr int NC  = NW * C;          // columns per block
    constexpr int RC  = 8;               // replayed tokens per shared-memory chunk

    __shared__ float qs[max_t][S_v];
    __shared__ float ks[max_t][S_v];
    __shared__ float vs[max_t][NC];
    __shared__ float gs[max_t];
    __shared__ float bs_[max_t];
    __shared__ float krs[RC][S_v];
    __shared__ float vrs[RC][NC];
    __shared__ float grs[RC];
    __shared__ float brs[RC];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp     = threadIdx.y;
    const int      tid      = warp * WARP_SIZE + lane;
    const int      col0     = blockIdx.z * NC + warp * C;

    const int64_t H_k   = a.qk_ne1;
    const int64_t D     = (int64_t) S_v * S_v * H;
    const int64_t F     = (int64_t) S_v * H_k + (int64_t) S_v * H + 2 * H;
    const int64_t off_v = (int64_t) S_v * H_k;
    const int64_t off_g = off_v + (int64_t) S_v * H;
    // chain: the new tokens are kept in the ring (their ring rows are written), state_out is the input state (no fold)
    // or the state after the replay (fold); otherwise state_out is the final state
    const bool    chain = n_len <= a.W;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = sequence;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    const int  p    = a.rep[2 * sequence + 0];
    const bool fold = a.rep[2 * sequence + 1] != 0;

    const float * cs = (a.state_rows ? a.state + (int64_t) a.state_rows[sequence] * a.state_row_stride
                                     : a.state + (int64_t) sequence * D) + (int64_t) h_idx * S_v * S_v;
    float * os = (a.out_rows ? a.out + (int64_t) a.out_rows[sequence] * a.out_row_stride
                             : a.out + (int64_t) sequence * D) + (int64_t) h_idx * S_v * S_v;

    // replay-only launch (long path) without rows to replay and without a state copy: nothing to do (the common case
    // of prompt ubatches)
    if (replay_only && p == 0 && !(chain && !fold && os != cs)) {
        return;
    }

    float st[C][R];
#pragma unroll
    for (int c = 0; c < C; ++c) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            st[c][r] = cs[(col0 + c) * S_v + r * WARP_SIZE + lane];
        }
    }

    // no fold: state_out is the input state (only a copy when it is not the source row itself)
    if (chain && !fold && os != cs) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                os[(col0 + c) * S_v + r * WARP_SIZE + lane] = st[c][r];
            }
        }
    }
    if (replay_only && p == 0) {
        return;
    }

    // all loads of the new tokens and of the first RC replayed tokens are issued before anything is stored to shared
    // memory (thread tid handles element tid of each 128-row, S_v == number of threads)
    static_assert(S_v == NW * WARP_SIZE && max_t * NC <= NW * WARP_SIZE && RC * NC <= NW * WARP_SIZE, "bad replay tiling");
    const int nr0 = min(RC, p);
    auto ring_row = [&](const int j) {
        return a.ring + (a.ring_rows ? (int64_t) a.ring_rows[sequence * a.R + j] : (int64_t) sequence * a.R + j) * F;
    };
    float qv_[max_t], kv_[max_t], kr_[RC];
#pragma unroll
    for (int t = 0; t < max_t; ++t) {
        if (t < n_tokens) {
            qv_[t] = a.q[iq3 * a.sq3 + t * a.sq2 + iq1 * a.sq1 + tid];
            kv_[t] = a.k[iq3 * a.sq3 + t * a.sq2 + iq1 * a.sq1 + tid];
        }
    }
#pragma unroll
    for (int j = 0; j < RC; ++j) {
        if (j < nr0) {
            kr_[j] = ring_row(j)[iq1 * S_v + tid];
        }
    }
    float v_one = 0.0f, vr_one = 0.0f, g_one = 0.0f, b_one = 0.0f, gr_one = 0.0f, br_one = 0.0f;
    const int tv = tid / NC, jv = tid % NC;
    if (tv < n_tokens) {
        v_one = a.v[sequence * a.sv3 + tv * a.sv2 + h_idx * a.sv1 + blockIdx.z * NC + jv];
    }
    if (tv < nr0) {
        vr_one = ring_row(tv)[off_v + h_idx * S_v + blockIdx.z * NC + jv];
    }
    if (tid < n_tokens) {
        const int64_t gb_offset = sequence * a.sb3 + tid * a.sb2 + h_idx * a.sb1;
        g_one = a.g[gb_offset];
        b_one = a.beta[gb_offset];
    }
    if (tid < nr0) {
        gr_one = ring_row(tid)[off_g + h_idx];
        br_one = ring_row(tid)[off_g + H + h_idx];
    }

#pragma unroll
    for (int t = 0; t < max_t; ++t) {
        if (t < n_tokens) {
            qs[t][tid] = qv_[t];
            ks[t][tid] = kv_[t];
        }
    }
#pragma unroll
    for (int j = 0; j < RC; ++j) {
        if (j < nr0) {
            krs[j][tid] = kr_[j];
        }
    }
    if (tv < n_tokens) {
        vs[tv][jv] = v_one;
    }
    if (tv < nr0) {
        vrs[tv][jv] = vr_one;
    }
    if (tid < n_tokens) {
        gs[tid]  = expf(g_one);
        bs_[tid] = b_one;
    }
    if (tid < nr0) {
        grs[tid] = expf(gr_one);
        brs[tid] = br_one;
    }
    __syncthreads();

    // l2 norms of the new q/k rows, one warp per token (independent of the replay, which overlaps their latency)
    if (a.l2) {
        for (int t = warp; t < n_tokens; t += NW) {
            float qv[R];
            float kv[R];
            float sq = 0.0f;
            float sk = 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                qv[r] = qs[t][r * WARP_SIZE + lane];
                kv[r] = ks[t][r * WARP_SIZE + lane];
                sq += qv[r] * qv[r];
                sk += kv[r] * kv[r];
            }
            sq = warp_reduce_sum<WARP_SIZE>(sq);
            sk = warp_reduce_sum<WARP_SIZE>(sk);
            const float fq = rsqrtf(sq / S_v + a.eps_q);
            const float fk = rsqrtf(sk / S_v + a.eps_k);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                qs[t][r * WARP_SIZE + lane] = a.scale_q * (fq * qv[r]);
                ks[t][r * WARP_SIZE + lane] = a.scale_k * (fk * kv[r]);
            }
        }
    }

    // replay, in chunks of RC ring rows (the first one is staged above)
    for (int c0 = 0; c0 < p; c0 += RC) {
        const int nr = min(RC, p - c0);
        if (c0 > 0) {
            __syncthreads();
            for (int e = tid; e < nr * S_v; e += NW * WARP_SIZE) {
                const int j = e / S_v;
                const int i = e % S_v;
                const float * row = a.ring + (a.ring_rows ? (int64_t) a.ring_rows[sequence * a.R + c0 + j]
                                                          : (int64_t) sequence * a.R + c0 + j) * F;
                krs[j][i] = row[iq1 * S_v + i];
            }
            for (int e = tid; e < nr * NC; e += NW * WARP_SIZE) {
                const int j = e / NC;
                const int jj = e % NC;
                const float * row = a.ring + (a.ring_rows ? (int64_t) a.ring_rows[sequence * a.R + c0 + j]
                                                          : (int64_t) sequence * a.R + c0 + j) * F;
                vrs[j][jj] = row[off_v + h_idx * S_v + blockIdx.z * NC + jj];
            }
            if (tid < nr) {
                const float * row = a.ring + (a.ring_rows ? (int64_t) a.ring_rows[sequence * a.R + c0 + tid]
                                                          : (int64_t) sequence * a.R + c0 + tid) * F;
                grs[tid] = expf(row[off_g + h_idx]);
                brs[tid] = row[off_g + H + h_idx];
            }
            __syncthreads();
        }

        for (int j = 0; j < nr; ++j) {
            float kr[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                kr[r] = krs[j][r * WARP_SIZE + lane];
            }
            const float gv = grs[j];
            const float bv = brs[j];
            float kv_shard[C];
#pragma unroll
            for (int c = 0; c < C; ++c) {
                kv_shard[c] = 0.0f;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    kv_shard[c] += st[c][r] * kr[r];
                }
            }
            const float kv_red = gdn_reduce_packed<C>(kv_shard, lane);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const float kv_col    = __shfl_sync(0xffffffff, kv_red, gdn_rs_lane<C>(c), WARP_SIZE);
                const float delta_col = (vrs[j][warp * C + c] - gv * kv_col) * bv;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    st[c][r] = gv * st[c][r] + kr[r] * delta_col;
                }
            }
        }
    }

    if (replay_only) {
        float * xs = a.scratch + (int64_t) sequence * D + (int64_t) h_idx * S_v * S_v;
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                xs[(col0 + c) * S_v + r * WARP_SIZE + lane] = st[c][r];
            }
        }
        return;
    }

    // fold: state_out is the state after the replay (not rewritten when that is the unchanged source row)
    if (chain && fold && !(p == 0 && os == cs)) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                os[(col0 + c) * S_v + r * WARP_SIZE + lane] = st[c][r];
            }
        }
    }

    // the normalized q/k rows of all warps
    __syncthreads();

    float * attn_data = a.attn + (sequence * n_tokens * H + h_idx) * S_v;

    for (int t = 0; t < n_tokens; ++t) {
        float kk[R];
        float qq[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            kk[r] = ks[t][r * WARP_SIZE + lane];
            qq[r] = qs[t][r * WARP_SIZE + lane];
        }
        const float gv = gs[t];
        const float bv = bs_[t];

        float kv_shard[C];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            kv_shard[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                kv_shard[c] += st[c][r] * kk[r];
            }
        }
        float attn_partial[C];
        const float kv_red = gdn_reduce_packed<C>(kv_shard, lane);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const float kv_col    = __shfl_sync(0xffffffff, kv_red, gdn_rs_lane<C>(c), WARP_SIZE);
            const float delta_col = (vs[t][warp * C + c] - gv * kv_col) * bv;
            attn_partial[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                st[c][r]         = gv * st[c][r] + kk[r] * delta_col;
                attn_partial[c] += st[c][r] * qq[r];
            }
        }
        const float attn_red = gdn_reduce_packed<C>(attn_partial, lane);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (lane == gdn_rs_lane<C>(c)) {
                attn_data[t * S_v * H + col0 + c] = attn_red * scale;
            }
        }
    }

    if (!chain) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                os[(col0 + c) * S_v + r * WARP_SIZE + lane] = st[c][r];
            }
        }
    }

    // ring rows of the new tokens
    for (int t = 0; t < (chain ? n_tokens : 0); ++t) {
        float * row = a.ring_out + (a.ring_out_rows ? (int64_t) a.ring_out_rows[sequence * n_tokens + t]
                                                    : (int64_t) sequence * n_tokens + t) * F;
        if (tid < NC) {
            row[off_v + h_idx * S_v + blockIdx.z * NC + tid] = vs[t][tid];
        }
        if (blockIdx.z == 0) {
            if (h_idx < H_k) {
                row[h_idx * S_v + tid] = ks[t][tid];
            }
            if (tid == 0) {
                row[off_g + h_idx]     = a.g[sequence * a.sb3 + t * a.sb2 + h_idx * a.sb1];
                row[off_g + H + h_idx] = bs_[t];
            }
        }
    }
}

bool ggml_cuda_gated_delta_net_replay_supported(const ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    return v->ne[0] == 128 && k->ne[0] == 128 && dst->src[3]->ne[0] == 1 && k->ne[3] == v->ne[3] &&
           v->ne[1] % k->ne[1] == 0 && ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size == WARP_SIZE &&
           v->ne[2] <= INT32_MAX;
}

bool ggml_cuda_op_gated_delta_net_replay(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_cuda_gdn_replay_args & a) {
    if (!ggml_cuda_gated_delta_net_replay_supported(dst)) {
        return false;
    }
    const ggml_tensor * src_v = dst->src[2];

    const int64_t S_v      = 128;
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    const uint3 neqk1_magic = init_fastdiv_values(a.qk_ne1);
    const float scale       = 1.0f / sqrtf((float) S_v);

    constexpr int C = GDN_REPLAY_COLS;
    const int num_warps = 4;
    const dim3 grid_dims(H, n_seqs, S_v / (num_warps * C));
    const dim3 block_dims(WARP_SIZE, num_warps, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    if (n_tokens <= 8) {
        auto launch = [&](auto MT) {
            ggml_cuda_kernel_launch(gated_delta_net_replay_cuda<decltype(MT)::value, C>, lp, a, H, (int) n_tokens, (int) n_tokens,
                                    neqk1_magic, scale, false);
        };
        if (n_tokens <= 1) {
            launch(std::integral_constant<int, 1>{});
        } else if (n_tokens <= 4) {
            launch(std::integral_constant<int, 4>{});
        } else {
            launch(std::integral_constant<int, 8>{});
        }
        return true;
    }

    // long batch: replay (with the arithmetic of the short kernel, which produced the ring rows of a verify batch)
    // into the scratch state, then the long kernel from there
    ggml_cuda_kernel_launch(gated_delta_net_replay_cuda<1, C>, lp, a, H, 0, (int) n_tokens, neqk1_magic, scale, true);

    gdn_seq_replay_params rp = {};
    rp.rep = a.rep;
    rp.x   = a.scratch;
    rp.out      = a.out;      rp.out_rows      = a.out_rows;      rp.out_row_stride = a.out_row_stride;
    rp.ring_out = a.ring_out; rp.ring_out_rows = a.ring_out_rows;
    rp.chain = n_tokens <= a.W ? 1 : 0;
    rp.H_k   = (int) a.qk_ne1;

    constexpr int P = 2, W = 4, T = 8;
    const dim3 grid_seq(H, n_seqs, S_v / (W * WARP_SIZE / P));
    const dim3 block_seq(WARP_SIZE, W, 1);
    const ggml_cuda_kernel_launch_params lps = ggml_cuda_kernel_launch_params(grid_seq, block_seq, 0, ctx.stream());
    const uint3 rq3_magic = init_fastdiv_values(1);
    ggml_cuda_kernel_launch(gated_delta_net_seq_cuda<P, W, T, false, true>, lps,
        a.q, a.k, a.v, a.g, a.beta, a.state, a.state_rows, a.state_row_stride, a.attn, (float *) nullptr, H, (int) n_tokens,
        a.sq1, a.sq2, a.sq3, a.sv1, a.sv2, a.sv3, a.sb1, a.sb2, a.sb3,
        neqk1_magic, rq3_magic, scale, a.l2 ? 1 : 0, a.eps_q, a.scale_q, a.eps_k, a.scale_k, (int64_t) 0, 1, rp);
    return true;
}

void ggml_cuda_op_gated_delta_net_replay_plain(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_q    = dst->src[0];
    const ggml_tensor * src_k    = dst->src[1];
    const ggml_tensor * src_v    = dst->src[2];
    const ggml_tensor * src_g    = dst->src[3];
    const ggml_tensor * src_beta = dst->src[4];
    const ggml_tensor * src_s    = dst->src[5];
    const ggml_tensor * src_ring = dst->src[6];
    const ggml_tensor * src_rep  = dst->src[7];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];
    const int64_t D        = S_v * S_v * H;

    ggml_cuda_gdn_replay_args a = {};
    a.q = (const float *) src_q->data; a.k = (const float *) src_k->data;
    a.sq1 = src_q->nb[1] / sizeof(float); a.sq2 = src_q->nb[2] / sizeof(float); a.sq3 = src_q->nb[3] / sizeof(float);
    a.qk_ne1 = src_k->ne[1];
    a.l2 = false;
    a.v = (const float *) src_v->data;
    a.sv1 = src_v->nb[1] / sizeof(float); a.sv2 = src_v->nb[2] / sizeof(float); a.sv3 = src_v->nb[3] / sizeof(float);
    a.g = (const float *) src_g->data; a.beta = (const float *) src_beta->data;
    a.sb1 = src_beta->nb[1] / sizeof(float); a.sb2 = src_beta->nb[2] / sizeof(float); a.sb3 = src_beta->nb[3] / sizeof(float);
    a.state = (const float *) src_s->data;
    a.ring  = (const float *) src_ring->data; a.R = (int) src_ring->ne[1];
    a.rep   = (const int32_t *) src_rep->data;
    a.attn     = (float *) dst->data;
    a.out      = a.attn + S_v * H * n_tokens * n_seqs;
    a.ring_out = a.out + D * n_seqs;
    a.W = ggml_get_op_params_i32(dst, 0);
    // the long path needs a scratch state: the pool
    ggml_cuda_pool_alloc<float> scratch(ctx.pool());
    if (n_tokens > 8) {
        a.scratch = scratch.alloc(D * n_seqs);
    }

    const bool ok = ggml_cuda_op_gated_delta_net_replay(ctx, dst, a);
    GGML_ASSERT(ok);
}

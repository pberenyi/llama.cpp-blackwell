#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

constexpr int gdn_cols_per_warp = 4;

// Step 3: helpers for the replay kernel (ported from the feveromo patch, 017-ggml_src_ggml-cuda_gated_delta_net.cu.patch)
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

// Step 2: replay kernel — identical token loop to gated_delta_net_cuda, but without the
// keep_rs_t snapshot block. No ring metadata yet; base state in, new tokens processed,
// final state out. For any batch this must produce bit-identical attn/state to the
// standard kernel (keep_rs_t=false path).
// Step 3: replay kernel with shared-memory staging (ported from the feveromo patch, 017-...gated_delta_net.cu.patch).
// Each warp owns C state columns of one head and reads them completely before writing them, so the state source may
// alias the state destination of the same sequence. The new tokens and (Step 4) the ring rows are staged in shared
// memory so they share the same processing path.
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

// Step 3 launcher: fixed S_v=128, C=GDN_REPLAY_COLS, dispatches on n_tokens
#ifndef GDN_REPLAY_COLS
#define GDN_REPLAY_COLS 4 // state columns per warp in the replay kernel
#endif

static void launch_gated_delta_net_replay(const ggml_cuda_gdn_replay_args & a,
        int64_t H, int64_t n_tokens, int64_t n_seqs, float scale, cudaStream_t stream) {
    constexpr int C = GDN_REPLAY_COLS;
    const int num_warps = 4;
    const dim3 grid_dims(H, n_seqs, 128 / (C * num_warps));
    const dim3 block_dims(32, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(a.qk_ne1);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    auto launch = [&](auto max_t) {
        ggml_cuda_kernel_launch(gated_delta_net_replay_cuda<decltype(max_t)::value, C>, launch_params,
            a, H, (int) n_tokens, (int) n_tokens, neqk1_magic, scale, false);
    };

    if (n_tokens <= 4) {
        launch(std::integral_constant<int, 4>{});
    } else if (n_tokens <= 8) {
        launch(std::integral_constant<int, 8>{});
    } else {
        GGML_ABORT("gated_delta_net_replay: n_tokens > 8 not supported in Step 3");
    }
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
    if (kda) {
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

void ggml_cuda_op_gated_delta_net_replay(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t , nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t , nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    GGML_ASSERT(S_v == 128);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;
    const float * s_d = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

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

    // recurrent state -> gdn_out tail (after attention scores)
    float * state_d = dst_d + S_v * H * n_tokens * n_seqs;

    // Step 3: p=0, no ring, no fold, no chain, no l2, no ring_out, no scratch
    // rep buffer: 2*n_seqs int32, all zero (p=0, fold=0)
    static int32_t * d_rep = nullptr;
    static int64_t  d_rep_cap = 0;
    if (2 * n_seqs > d_rep_cap) {
        if (d_rep) cudaFree(d_rep);
        d_rep_cap = 2 * n_seqs;
        cudaMalloc(&d_rep, d_rep_cap * sizeof(int32_t));
    }
    cudaMemsetAsync(d_rep, 0, 2 * n_seqs * sizeof(int32_t), stream);

    ggml_cuda_gdn_replay_args a = {};
    a.q = q_d;  a.k = k_d;
    a.sq1 = sq1; a.sq2 = sq2; a.sq3 = sq3;
    a.qk_ne1 = neqk1;
    a.l2 = false;
    a.eps_q = 0.0f; a.scale_q = 1.0f; a.eps_k = 0.0f; a.scale_k = 1.0f;
    a.v = v_d;  a.sv1 = sv1; a.sv2 = sv2; a.sv3 = sv3;
    a.g = g_d;  a.beta = b_d; a.sb1 = sb1; a.sb2 = sb2; a.sb3 = sb3;
    a.state = s_d; a.state_rows = nullptr; a.state_row_stride = 0;
    a.ring = nullptr; a.ring_rows = nullptr; a.R = 0;
    a.rep = d_rep;
    a.attn = dst_d;
    a.out = state_d; a.out_rows = nullptr; a.out_row_stride = 0;
    a.ring_out = nullptr; a.ring_out_rows = nullptr;
    a.scratch = nullptr;
    a.W = 0;

    launch_gated_delta_net_replay(a, H, n_tokens, n_seqs, scale, stream);
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const char * env = getenv("LLAMA_GDN_REPLAY");
    if (env && env[0] == '1') {
        ggml_cuda_op_gated_delta_net_replay(ctx, dst);
        return;
    }
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);


// inputs of the short-sequence kernel when q/k normalization and/or the state gather are fused into it
struct ggml_cuda_gdn_short_args {
    const float * q;             // q (raw when l2), element (i, h, t, s) at q + s*sq3 + t*sq2 + h*sq1 + i
    const float * k;             // k, same strides as q
    int64_t sq1, sq2, sq3;
    int64_t qk_ne1, qk_ne3;      // q/k head count and sequence count
    bool  l2;                    // normalize q/k: x * scale / sqrt(mean(x^2) + eps)
    float eps_q, scale_q, eps_k, scale_k;
    const float   * state;       // initial state (row 0) or cache base when state_rows != nullptr
    const int32_t * state_rows;  // optional device row index into `state` (single sequence)
    int64_t state_row_stride;    // floats
};

// returns false if the shape is not supported by the short kernel
bool ggml_cuda_op_gated_delta_net_short(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gdn_short_args & args, const ggml_cuda_gated_delta_net_fused_cache * cache);

struct ggml_cuda_gdn_gates_args {
    const void  * wa; int64_t wa_s1;  // BF16 alpha weight [ncols, nrows], row stride in elements
    const void  * wb; int64_t wb_s1;  // BF16 beta weight
    const float * x;  int64_t x_s1;   // activations [ncols, n_t]
    const float * dt;                 // [nrows]
    const float * a;                  // [nrows]
    float * g; int64_t g_s1;          // softplus(Wa x + dt) * a   [nrows, n_t]
    float * b; int64_t b_s1;          // sigmoid(Wb x)              [nrows, n_t]
    int nrows, ncols, n_t;
};

bool ggml_cuda_op_gdn_gates(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_gates_args & args);

// long-sequence kernel (S_v = 128, n_tokens > 8); returns false if not applicable
// args: optional fused q/k normalization and state gather (nullptr: plain op inputs)
bool ggml_cuda_op_gated_delta_net_seq(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                      const ggml_cuda_gdn_short_args * args, const ggml_cuda_gated_delta_net_fused_cache * cache);

// gated delta net with rollback by replay (GGML_OP_GATED_DELTA_NET_REPLAY), S_v = S_k = 128, scalar gate.
// Row-index pointers are optional: nullptr means the plain op tensor layout (see ggml_gated_delta_net_replay).
struct ggml_cuda_gdn_replay_args {
    // new tokens: q/k are raw when l2 (normalized in the kernel), else already normalized
    const float * q; const float * k;
    int64_t sq1, sq2, sq3;
    int64_t qk_ne1;                   // H_k
    bool  l2; float eps_q, scale_q, eps_k, scale_k;
    const float * v; int64_t sv1, sv2, sv3;
    const float * g; const float * beta; int64_t sb1, sb2, sb3;
    // initial state: state_rows ? state + state_rows[s]*state_row_stride : state + s*D
    const float * state; const int32_t * state_rows; int64_t state_row_stride;
    // replayed tokens: ring_rows ? ring + ring_rows[s*R + t]*F : ring + (s*R + t)*F; rep[2*s] = number of tokens,
    // rep[2*s + 1] = fold flag
    const float * ring; const int32_t * ring_rows; int R;
    const int32_t * rep;
    // outputs
    float * attn;
    float * out;      const int32_t * out_rows;      int64_t out_row_stride; // state_out
    float * ring_out; const int32_t * ring_out_rows;                         // row s*n_tokens + t
    float * scratch;  // D*n_seqs floats (post-replay state of the long path)
    int W;
};

// returns false if the shape is not supported
bool ggml_cuda_op_gated_delta_net_replay(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_cuda_gdn_replay_args & args);

// plain op (inputs and outputs as in the op tensors)
bool ggml_cuda_gated_delta_net_replay_supported(const ggml_tensor * dst);
void ggml_cuda_op_gated_delta_net_replay_plain(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

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

// Step 3: replay kernel with shared-memory staging (new tokens + ring rows), p=0 for now.
// Row-index pointers are optional: nullptr means the plain op tensor layout.
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

// returns false if the shape is not supported by the replay kernel
void ggml_cuda_op_gated_delta_net_replay(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1);
    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const int64_t n_seqs = ubatch.n_seqs;

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    build_conv_state_store(inp, conv_states_all, conv_input, conv_kernel_size - 1, 0);

    return conv_input;
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_rs_replay(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t size   = mctx_cur->get_size();
    const int64_t head   = mctx_cur->get_head();
    const int32_t rs_z   = mctx_cur->get_rs_z();
    const int64_t Cap    = mctx_cur->get_rs_Cap();
    const int64_t D      = hparams.n_embd_s();

    ggml_tensor * ring_all = mctx_cur->get_q_l(il);
    const int64_t F        = ring_all->ne[0];

    GGML_ASSERT(ssm_states_all->ne[0] == D && ssm_states_all->ne[1] == size);

    // a new sequence starts from the row of rs_z (zeroed; no cell reads it otherwise)
    ggml_tensor * state_zero = ggml_view_1d(ctx0, ssm_states_all, D*(rs_z >= 0), rs_z*ssm_states_all->nb[1]*(rs_z >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // gathers first: the moves below may overwrite source rows
    ggml_tensor * states = ggml_get_rows(ctx0, ssm_states_all, inp->rp_s_src);
    ggml_build_forward_expand(gf, states);

    ggml_tensor * ring = ggml_get_rows(ctx0, ring_all, inp->rp_ring_src);
    ring = ggml_reshape_3d(ctx0, ring, F, Cap, n_seqs);
    ggml_build_forward_expand(gf, ring);

    if (n_rs > n_seqs) {
        // the other cells in the range move with their state and their whole ring buffer
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_get_rows(ctx0, ssm_states_all, inp->rp_extra),
            ggml_view_2d(ctx0, ssm_states_all, D, n_rs - n_seqs, ssm_states_all->nb[1], (head + n_seqs)*ssm_states_all->nb[1])));

        ggml_tensor * ring_cells = ggml_reshape_2d(ctx0, ring_all, F*Cap, size);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_get_rows(ctx0, ring_cells, inp->rp_extra),
            ggml_view_2d(ctx0, ring_cells, F*Cap, n_rs - n_seqs, ring_cells->nb[1], (head + n_seqs)*ring_cells->nb[1])));
    }

    return { states, ring };
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il,
        ggml_tensor *        ring) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0 && ring == nullptr;

    if (inp->rp_idx != nullptr && n_seq_tokens <= mctx_cur->get_rs_W()) {
        // rollback by replay: one state per sequence plus the inputs of the tokens after it, instead of
        // 1 + n_rs_seq snapshots
        GGML_ASSERT(ring != nullptr);
        ggml_tensor * ring_all = mctx_cur->get_q_l(il);

        const int64_t W     = mctx_cur->get_rs_W();
        const bool    chain = n_seq_tokens <= W;
        const int64_t D     = S_v * S_v * H_v;
        const int64_t F     = ring_all->ne[0];

        ggml_tensor * out = ggml_gated_delta_net_replay(ctx0, q, k, v, g, b, s, ring, inp->rp_rep, (int32_t) W);
        if (n_seq_tokens > 1) {
            res->add_fused_node({LLM_FUSED_OP_GDN_CH, out, il});
        } else {
            res->add_fused_node({LLM_FUSED_OP_GDN_AR, out, il});
        }

        const size_t es      = ggml_element_size(out);
        const size_t off_fin = (size_t) (S_v * H_v * W * n_seqs) * es;

        ggml_tensor * output = ggml_view_4d(ctx0, out, S_v, H_v, n_seq_tokens, n_seqs,
            S_v * es, S_v * H_v * es, S_v * H_v * n_seq_tokens * es, 0);
        cb(output, "attn_output", il);


        ggml_tensor * state_out = ggml_view_2d(ctx0, out, D, n_seqs, D * es, off_fin);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, ssm_states_all, state_out, inp->rp_s_dst));

        if (chain) {
            ggml_tensor * rnew = ggml_view_2d(ctx0, out, F, n_seq_tokens * n_seqs, F * es, off_fin + D * n_seqs * es);
            ggml_tensor * ring_dst = ggml_view_1d(ctx0, inp->rp_ring_dst, n_seq_tokens * n_seqs, 0);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, ring_all, rnew, ring_dst));
        }

        return output;
    }

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K);
    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
        D, n_seqs, n_written,
        ssm_states_all->nb[1],
        (size_t) mem_size * row_size,
        (size_t) kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

    return output;
}

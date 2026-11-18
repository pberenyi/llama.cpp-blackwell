#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                         bool   no_alloc,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    // rollback by replay (delta-net models, n_rs_seq > 0): instead of 1 + n_rs_seq state snapshots, s_l keeps one state
    // per cell (a "base") and q_l a circular buffer of rs_Cap rows per cell with the inputs of the tokens processed
    // since that base (pending rows). A graph replays the pending rows, processes its tokens and appends their rows;
    // the base is only rewritten ("fold": the replayed rows are folded into it) when the buffer would overflow, when
    // the cell moves, and for ubatches of more than rs_W tokens (which store their final state and keep no rows).
    // A rollback only drops pending rows (GGML_OP_GATED_DELTA_NET_REPLAY). The conv state (r_l) keeps its
    // 1 + n_rs_seq snapshots (rs_idx).
    bool     rs_replay = false;
    uint32_t rs_W      = 0; // longest ubatch (tokens per sequence) whose rows are kept (n_rs_seq + 1)
    uint32_t rs_Cap    = 0; // ring rows per cell
    int64_t  rs_F      = 0; // ring row size (floats)

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        // rollback by replay (rs_replay): first pending ring row, number of pending rows, rows appended by the last
        // ubatch (the most that a rollback can drop)
        uint32_t  off    = 0;
        uint32_t  cnt    = 0;
        uint32_t  last_n = 0;
        uint32_t  off0   = 0; // like src0: the values before the current ubatch and its fold decision, for the inputs
        uint32_t  cnt0   = 0;
        uint32_t  fold0  = 0;

        // largest rollback whose snapshot the last ubatch guarantees: a ubatch of n <= n_rs_seq + 1 tokens
        // writes snapshots 0..n-1, a longer one only guarantees the final state (ggml_gated_delta_net)
        uint32_t rs_max = 0;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;
    // ring of replay inputs (rs_replay)
    std::vector<ggml_tensor *> q_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the RS cache along with the allocated backend buffers:
    llama_memory_bufs ctxs_bufs;

    // true if no layers - can happen if the layer filter removes all layers
    bool is_empty() const;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    // cell_ranges: rows of r_l; cell_ranges_s: rows of s_l (default: the same); reps: per written cell, the ring row of
    // its first pending row (cell*rs_Cap + off) and the number of pending rows (rs_replay)
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges,
                          const std::vector<std::pair<uint32_t, uint32_t>> * cell_ranges_s = nullptr,
                          const std::vector<std::pair<uint32_t, uint32_t>> * reps = nullptr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;
    ggml_tensor * get_q_l(int32_t il) const;

    int32_t s_copy(int i) const;

    // rollback by replay
    bool     is_rs_replay() const;
    uint32_t get_rs_W()     const;
    uint32_t get_rs_Cap()   const;
    int64_t  get_rs_F()     const;

    // number of I32 values of the replay index input for a ubatch
    int64_t rs_replay_n_idx(uint32_t n_seqs, uint32_t n_seq_tokens) const;
    // layout: [s_src n_seqs | rep 2*n_seqs (count, fold) | ring_src Cap*n_seqs | s_dst n_seqs |
    //          ring_dst n_seq_tokens*n_seqs if n_seq_tokens <= W | extra (n_rs - n_seqs)]
    void set_input_rs_replay(int32_t * data, uint32_t n_seqs, uint32_t n_seq_tokens) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};

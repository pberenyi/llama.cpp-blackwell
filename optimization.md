# Porting the feveromo 5070 Ti optimizations onto current llama.cpp

**Target:** Qwen 3.8 27B (GSQ-RCO IQ3_S + MTP head) on RTX 5070 Ti (16 GB, sm_120)
**Goal:** recover most of the feveromo patch's ~1.9× decode speedup (stock ~77 → patched ~147 tok/s)
**on top of current llama.cpp**, as a maintainable, per-component port — not a pinned fork.

**Source patch:** `D:\install\ai\recipes-qwen3.8-27b-5070ti\cuda-kernels.patch`
(pinned to llama.cpp commit `4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0`)

**Known-good baseline:** stock build of `4b1a27fa0` with the optimized CMake flags →
**70–80 tok/s measured** (vs 40–50 tok/s for downloadable release binaries).

---

## 0. Why the full patch cannot be applied to this tree

1. **MSVC incompatibility (confirmed).** The patch's `mmvq.cu` template dispatch
   (`mul_mat_vec_q_switch_fusion` / `mul_mat_vec_q_switch_ncols_dst`) fails under
   MSVC's constant evaluator: `error C2672 ... expression did not evaluate to a
   constant`. This is a g++-permissive/MSVC-strict template pattern; no CMake or
   compiler flag fixes it.
2. **The pattern exists in current llama.cpp too.** `mmvq.cu` in *this* tree has
   the same `mul_mat_vec_q_switch_fusion` at line 1047 and the same
   `mul_mat_vec_q_switch_ncols_dst` dispatch. So the MSVC failure is **independent
   of the patch** — it would occur on any build of this tree under MSVC if that
   dispatch path is instantiated the same way. (Stock builds of this tree
   presumably compile because the *patched* instantiation context — extra
   template parameters / fusion args — is what trips the constant evaluator.)
3. **Tight coupling.** The patch's mmvq API changes are coupled into
   `ggml-cuda.cu` (7-arg `ggml_cuda_mul_mat_vec_q`, `ggml_cuda_mmvq_s8_can_fuse_mixed`,
   `ggml_cuda_mmvq_s8_usable`). Reverting mmvq alone breaks `ggml-cuda.cu` —
   selective revert is not separable.

**Consequence:** port components individually, in value order, and **skip the
mmvq/mmvf/mmq GEMM micro-optimizations entirely**.

---

## 1. Component ranking (value × portability ÷ risk)

| Rank | Component | Patch files | Value | Portability to this tree |
|------|-----------|-------------|-------|--------------------------|
| 1 | **Q4 KV flash attention** | `fattn-mma-q4i8.cu/.cuh` (new) | High | **High** — files absent here, clean add |
| 2 | **Gumbel / coupled sampling** | `gumbel.cu/.cuh` (new) | Medium | **High** — files absent here, clean add |
| 3 | **MTP verify path rewrite** | `speculative.cpp`, `gated_delta_net.cu` | **Highest** (~10.6→5.2 ms/verify token) | **Medium** — latest already has a full MTP engine; port = optimization of existing code, needs API mapping |
| 4 | **GDN state rollback (576→156 MiB)** | `llama-memory-recurrent.cpp/.h` | High (enables MTP at 128K on 16 GB) | **Medium** — file exists here; discrete memory-layout change |
| 5 | **Model graphs** | `qwen35.cpp`, `qwen35moe.cpp`, `delta-net-base.cpp` | — | **Skip** — all three already present in this tree |
| 6 | **mmvq/mmvf/mmq GEMM micro-opts** | `mmvq.cu/.cuh`, `mmvf.cu`, `mmq.*` | Low–Medium | **Skip** — MSVC-incompatible + coupled into `ggml-cuda.cu` |
| 7 | **CPU/core/misc glue** | `ggml.c`, `ggml-cpu/*`, `common/*`, `llama-ext.h` | Low | Port only as required by ranks 1–4 |

Verification of this tree (2026-11-02):

- `ggml/src/ggml-cuda/fattn-mma-q4i8.cu` — **not present**
- `ggml/src/ggml-cuda/gumbel.cu` — **not present** (no gumbel match in `unary.cu` either)
- `common/speculative.cpp` — **present**, 3,170 lines, full `common_speculative_impl_draft_mtp`
  (including `chain_heads` mode for qwen35/qwen35moe)
- `src/models/qwen35.cpp`, `qwen35moe.cpp`, `delta-net-base.cpp` — **present**
- `src/llama-memory-recurrent.cpp` — **present** (48 KB)
- `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` — **present** (Blackwell MMQ already upstream)
- `docs/speculative.md` — present (MTP documented upstream)

---

## 2. Phase 0 — Stock baseline of this tree

Build current llama.cpp with the known-good Windows flags (x64 Native Tools
prompt, CUDA 13.4):

```powershell
cmake -S . -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_SYSTEM_PROCESSOR=AMD64 `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc" `
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4" `
  -DCMAKE_CUDA_ARCHITECTURES=120a-real `
  -DCMAKE_CUDA_FLAGS="--use_fast_math" `
  -DGGML_CUDA=ON -DGGML_NATIVE=ON `
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=ON `
  -DGGML_CUDA_FA_ALL_QUANTS=OFF `
  '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16' `
  -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_FORCE_MMQ=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF `
  -DGGML_CPU=ON -DGGML_CPU_REPACK=ON -DGGML_OPENMP=ON `
  -DGGML_BLAS=OFF -DGGML_LTO=OFF -DGGML_BACKEND_DL=OFF `
  -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=ON

cmake --build build --config Release --target llama-server --parallel 8
```

**Gate:** builds clean, runs at ~70–80 tok/s on the IQ3_S model with
`--ctx-size 131072 --cache-type-k q4_0 --cache-type-v q4_0`.
Record the number — it's the baseline for every later benchmark.

---

## 3. Phase 1 — New files: Q4 KV FA + Gumbel (lowest risk)

Copy the four new files from the patched tree
(`D:\install\ai\llama.cpp-pi2\ggml\src\ggml-cuda\`) into
`D:\install\ai\llama.cpp-opt\ggml\src\ggml-cuda\`:

```
fattn-mma-q4i8.cu
fattn-mma-q4i8.cuh
gumbel.cu
gumbel.cuh
```

Then:

1. **Wire into `ggml/src/ggml-cuda/CMakeLists.txt`** — add the four files to the
   source list (match how `fattn-mma-f16.cuh` / `unary.cu` are listed).
2. **Check the include/registration surface.** The new files must be referenced
   from `fattn.cu` (kernel selection for Q4_0 KV) and `unary.cu`/`ggml-cuda.cu`
   (Gumbel op dispatch). In the patched tree these references live in the
   *patched* versions of those files. For this port, take **only the specific
   hunks** in `fattn.cu` / `unary.cu` / `ggml-cuda.cu` that
   (a) `#include` the new headers and (b) add the kernel dispatch entries —
   not the whole-file patch. Extract them from
   `cuda-kernels.patch` (search for `fattn-mma-q4i8` and `gumbel` in the diff).
3. **Build.**

**Gate:** compiles. The new files are self-contained CUDA kernels; if they
reference symbols that moved between `4b1a27fa0` and current master, the
compiler will name them — fix each reference individually.

**Expected result:** Q4_0 KV flash attention + coupled Gumbel sampling
available. No speed change yet (nothing uses them until phases 3–4 are in),
but the FA kernel is a prerequisite for the 128K KV budget.

---

## 4. Phase 2 — MTP verify path rewrite (highest value, hardest port)

Latest `common/speculative.cpp` already contains a complete MTP engine:
`common_speculative_impl_draft_mtp` with three modes (mem-shared, chain-heads,
single-head — qwen35 explicitly supported). The patch's contribution is **not
the engine** — it's the **verify-path optimization** in
`gated_delta_net.cu` (~10.6 → 5.2 ms per verify token) plus supporting
changes in `speculative.cpp`.

Steps:

1. **Diff the two engines.** Extract the patch hunks for `speculative.cpp`
   and `gated_delta_net.cu` from `cuda-kernels.patch`. Read them against this
   tree's `common/speculative.cpp` (3,170 lines) and
   `ggml/src/ggml-cuda/gated_delta_net.cu` (17 KB) to identify:
   - which hunks are **API-compatible** as-is (same function signatures),
   - which need **mapping** onto the current engine's structure
     (`params.ctx_dft`, `n_mtp_layers`, `chain_heads`, `is_mem_shared`),
   - which are **superseded** by current upstream work (discard).
2. **Port the GDN verify kernel first** (`gated_delta_net.cu`) — it is a
   self-contained CUDA kernel rewrite. Check this tree's
   `gated-delta-net-mma.cu` (19 KB) — the current tree has an MMA GDN
   variant the pinned commit did not; make sure the ported kernel coexists
   (or supersedes) it consistently.
3. **Port the speculative.cpp hunks second**, adapting to the current
   `common_speculative_impl_draft_mtp` structure.
4. **Check `llama-ext.h`** — the current tree already has the staging API
   (`llama_set_embeddings_nextn` / `llama_get_embeddings_nextn_ith`, 8 KB file
   present); the patch's `llama-ext.h` changes may be redundant.
5. **Build + smoke test** MTP at short context (`--ctx-size 8192`) before
   pushing to 128K.

**Gate:** MTP runs (`--spec-type draft-mtp --spec-draft-n-max 5
--spec-draft-adaptive --spec-coupled-sampling`) without crashes, and
per-verify-token cost is measurably lower than stock (watch the server's
draft acceptance / spec stats).

**Risk:** this is the phase most likely to hit real API drift. Budget for
manual adaptation, not copy-paste.

---

## 5. Phase 3 — GDN recurrent state rollback (576 → 156 MiB)

This tree has `src/llama-memory-recurrent.cpp` (48 KB). The patch shrinks the
Gated DeltaNet recurrent state that must be snapshotted/rolled back per
speculative step from 576 MiB to 156 MiB — the difference between MTP
fitting in 16 GB at 128K and not.

Steps:

1. Extract the `llama-memory-recurrent.cpp/.h` hunks from the patch.
2. Compare the state-layout code in this tree against the pinned commit.
   If the current tree already restructured recurrent state (the file grew
   substantially), adapt; if the structure matches, apply directly.
3. **Build + verify:** at 128K context with MTP, check peak VRAM in
   `nvidia-smi` / server logs — the recurrent-state contribution should be
   ~156 MiB, not 576 MiB.

**Gate:** MTP + 128K loads and serves on 16 GB (the whole point of this
component).

---

## 6. Phase 4 — Model graphs: SKIP (verified)

`qwen35.cpp`, `qwen35moe.cpp`, and `delta-net-base.cpp` all exist in this
tree. The patch's model-graph changes were written against `4b1a27fa0`, which
predates current model support. **Do not port.** If the model fails to load,
investigate the GGUF ↔ current model API compatibility directly — that is a
separate, more likely upstream issue than a missing graph.

---

## 7. What to skip, and why

| Skipped | Reason |
|---------|--------|
| `mmvq.cu/.cuh`, `mmvf.cu` GEMM micro-opts | MSVC C2672 constant-eval failure (confirmed on the pinned commit); coupled into `ggml-cuda.cu` API (7-arg `ggml_cuda_mul_mat_vec_q`); the same dispatch pattern exists in this tree's `mmvq.cu` line 1047 |
| `mmq.cu/.cuh`, `mmq-config-*.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh` | Blackwell MMQ already upstream (`mmq-config-blackwell.cuh` present); patch's mmq work targets the old commit's MMQ |
| `ggml-cuda.cu` (whole file) | Central dispatch; its patch hunks are dominated by the mmvq API coupling that we're skipping |
| `qwen35.cpp`, `qwen35moe.cpp`, `delta-net-base.cpp` | Already in current tree |
| `common/arg.cpp`, `common/common.h`, `common/sampling.cpp/.h` (whole-file) | Port only the specific hunks needed by phases 1–3 (e.g. `--spec-draft-vocab`, `--spec-coupled-sampling` args if not already upstream) |
| `tests/*`, `tools/server/*` | Optional; port `test-sampling-coupled.cpp` / `test-gdn-replay.cpp` only if useful for verification |

---

## 8. Per-phase workflow (repeat for each phase)

1. `git status` clean, note the current commit.
2. Apply the phase's changes (copy files / targeted hunks).
3. `cmake --build build --config Release --target llama-server --parallel 8`.
4. On failure: fix the **named** symbol/line; do not revert the whole phase
   unless the component is structurally incompatible.
5. Run the phase gate (section 3–5 above).
6. Commit the phase as a separate git commit (e.g.
   `feat: port Q4 KV flash attention from feveromo-5070ti patch`).

Keeping phases as separate commits means any single phase can be
`git revert`-ed later without touching the others — this is the
maintainability win over the monolithic patch.

---

## 9. Benchmark plan (after each phase)

Same protocol each time, so numbers are comparable:

```powershell
# decode: 256 tokens, short context
.\build\Release\llama-server.exe -m <model.gguf> --ctx-size 8192 --n-gpu-layers all `
  --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 `
  --batch-size 512 --ubatch-size 256 --threads 8 --threads-batch 8 --parallel 1 `
  --jinja <phase-specific spec flags>
```

| Checkpoint | Metric | Target |
|------------|--------|--------|
| Phase 0 (stock) | tok/s @ 8K | ~70–80 (baseline) |
| Phase 1 (FA + Gumbel) | tok/s @ 8K | ≈ baseline (infra only) |
| Phase 2 (MTP verify) | tok/s @ 8K, draft acceptance | ↑, acceptance stable |
| Phase 3 (state rollback) | loads at 128K + MTP; tok/s @ 128K | loads; ~1.3–1.6× baseline |
| Final | tok/s @ 8K / 64K / 128K | compare vs recipe's 147 @ 128K |

Record results in `benchmarks/<date>-<phase>.json` (the recipe repo's
`benchmark.py` / `benchmark_chat.py` are reusable for this).

---

## 10. Stop conditions

Abandon the native port (fall back to WSL2 for the full patch, or keep the
stock build) if:

- **Phase 2 (MTP)** requires rewriting more than ~30% of the current
  `speculative.cpp` MTP impl — at that point you're rebuilding upstream's
  engine, not porting an optimization.
- **Phase 3 (state rollback)** conflicts with a current-tree redesign of
  `llama-memory-recurrent.cpp` that can't be reconciled in an afternoon.
- Any phase produces **numerical quality regressions** (garbage output,
  divergent draft acceptance) that can't be traced to a specific hunk.

---

## 11. Status log

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| 2026-11-02 | — | Plan created | Tree verified: q4i8 FA, Gumbel absent; MTP engine + qwen35 models + Blackwell MMQ present |
| | 0 | pending | |
| | 1 | pending | |
| | 2 | pending | |
| | 3 | pending | |

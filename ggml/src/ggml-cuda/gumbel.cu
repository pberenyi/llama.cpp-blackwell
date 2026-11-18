#include "gumbel.cuh"

#include <type_traits>

// keep in sync with the CPU version (ggml_gumbel_noise in ggml.c, ggml_compute_forward_add_gumbel in ggml-cpu/ops.cpp)

#define GUMBEL_MAX_CAND 128

static __device__ __forceinline__ uint64_t gumbel_mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// the double math keeps the result equal to the CPU one (also with -use_fast_math)
static __device__ __forceinline__ float gumbel_noise(uint32_t seed, uint64_t ctr, int32_t id) {
    const uint64_t key = gumbel_mix64(gumbel_mix64((uint64_t) seed + 0x9e3779b97f4a7c15ULL) ^ ctr);
    const uint64_t r   = gumbel_mix64(key + ((uint64_t) (uint32_t) id + 1)*0x9e3779b97f4a7c15ULL);

    const double u = ((double) (r >> 12) + 0.5)*0x1p-52;

    return (float) -log(-log(u));
}

// float -> uint32 with the same order
static __device__ __forceinline__ uint32_t gumbel_key(float x) {
    const uint32_t b = __float_as_uint(x);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

struct gumbel_sel {
    int32_t  all;  // keep all of [0, n_valid)
    uint32_t lkey; // else keep the elements up to (lkey, lidx) in the order (value desc, index asc)
    int32_t  lidx;
};

static __device__ __forceinline__ float gumbel_key_to_float(uint32_t key) {
    return __uint_as_float((key & 0x80000000u) ? (key ^ 0x80000000u) : ~key);
}

// one block per row: find which elements top-k/top-p keep
// E > 0: the keys of the row stay in registers (n <= E*nthreads), E == 0: read the row in each pass
template <int nthreads, int E>
__launch_bounds__(nthreads, 1)
static __global__ void k_gumbel_select(const float * __restrict__ a, const int32_t * __restrict__ par, gumbel_sel * __restrict__ sel,
        const int64_t n, const int64_t s01) {
    const int64_t ir   = blockIdx.x;
    const int     tid  = threadIdx.x;
    const int     lane = tid % WARP_SIZE;
    const int     warp = tid / WARP_SIZE;

    const float * x = a + ir*s01;

    const float   t     = __int_as_float(par[2]);
    const int64_t nv    = par[3] > 0 ? min((int64_t) par[3], n) : n;
    const int     top_k = par[4];
    const float   top_p = __int_as_float(par[5]);

    if (t == 0.0f || top_k <= 0 || nv <= 0) {
        if (tid == 0) {
            sel[ir] = { 1, 0, 0 };
        }
        return;
    }

    constexpr int nwarps   = nthreads/WARP_SIZE;
    constexpr int unroll   = 8;
    constexpr int max_gath = GUMBEL_MAX_CAND; // candidates gathered for the exact selection

    __shared__ uint32_t s_red[2][WARP_SIZE];
    __shared__ int      s_ng;
    __shared__ int      s_gidx[max_gath];
    __shared__ uint32_t s_gkey[max_gath];
    __shared__ int      s_sorted[max_gath];
    __shared__ float    s_exp[GUMBEL_MAX_CAND];

    // keys of elements tid + e*nthreads, 0 past nv (0 is below every threshold used for counting)
    uint32_t keys[E > 0 ? E : 1];
    if constexpr (E > 0) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int i = e*nthreads + tid;
            keys[e] = i < nv ? gumbel_key(x[i]) : 0;
        }
    }

    // call f(i, key) for the elements of this thread (with_idx = false: i is not valid and elements past nv have key 0)
    auto for_each = [&](auto && f, auto with_idx) {
        if constexpr (E > 0) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                if constexpr (decltype(with_idx)::value) {
                    const int i = e*nthreads + tid;
                    if (i < nv) {
                        f(i, keys[e]);
                    }
                } else {
                    f(0, keys[e]);
                }
            }
        } else {
            for (int i0 = 0; i0 < nv; i0 += nthreads*unroll) {
                // several loads in flight per thread
                float xv[unroll];
#pragma unroll
                for (int u = 0; u < unroll; ++u) {
                    const int i = i0 + u*nthreads + tid;
                    xv[u] = i < nv ? x[i] : 0.0f;
                }
#pragma unroll
                for (int u = 0; u < unroll; ++u) {
                    const int i = i0 + u*nthreads + tid;
                    if (i < nv) {
                        f(i, gumbel_key(xv[u]));
                    }
                }
            }
        }
    };
    using no_idx = std::integral_constant<bool, false>;
    using idx    = std::integral_constant<bool, true>;

    // block-wide sum / max of one value per thread (s_red is double buffered: one barrier per call)
    int red_buf = 0;
    auto warp_reduce = [&](uint32_t v, bool is_max) -> uint32_t {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        return is_max ? __reduce_max_sync(0xFFFFFFFF, v) : __reduce_add_sync(0xFFFFFFFF, v);
#else
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            const uint32_t o = __shfl_xor_sync(0xFFFFFFFF, v, off, WARP_SIZE);
            v = is_max ? max(v, o) : v + o;
        }
        return v;
#endif
    };
    auto block_reduce = [&](uint32_t v, bool is_max) -> uint32_t {
        v = warp_reduce(v, is_max);
        if (lane == 0) {
            s_red[red_buf][warp] = v;
        }
        __syncthreads();
        v = warp_reduce(lane < nwarps ? s_red[red_buf][lane] : 0, is_max);
        red_buf ^= 1;
        return v;
    };

    auto count_ge = [&](uint32_t th) -> uint32_t {
        uint32_t c = 0;
        for_each([&](int, uint32_t key) {
            c += key >= th;
        }, no_idx{});
        return block_reduce(c, false);
    };

    const uint32_t kk = (uint32_t) min((int64_t) top_k, nv);

    // find a threshold lo <= (top_k-th largest key) with at most max_gath elements >= lo
    // first try 10 below the max value, then bisect the keys
    uint32_t mx = 0;
    for_each([&](int, uint32_t key) {
        mx = max(mx, key);
    }, no_idx{});
    mx = block_reduce(mx, true);

    uint32_t lo = 0;      // count(lo) >= kk
    uint32_t hi = mx + 1; // count(hi) <  kk (count(mx + 1) = 0), unless mx is the largest key
    uint32_t c_lo = (uint32_t) nv;
    {
        const float    vmx = gumbel_key_to_float(mx);
        const uint32_t th  = gumbel_key(fmaxf(vmx - 10.0f, -FLT_MAX));
        const uint32_t c   = count_ge(th);
        if (c >= kk) {
            lo   = th;
            c_lo = c;
        } else {
            hi = th;
        }
    }
    // (a few elements more than kk are cheaper to sort than more bisection steps)
    const uint32_t c_target = min((uint32_t) max_gath, max(kk, 32u));
    while (c_lo > c_target && hi - lo > 1) {
        // the middle value (fewer steps than the middle key); any key in (lo, hi) keeps the invariants
        const float lv = gumbel_key_to_float(lo);
        const float hv = gumbel_key_to_float(hi);
        uint32_t mid = gumbel_key(lv + 0.5f*(hv - lv));
        if (!(mid > lo && mid < hi)) {
            mid = lo + (hi - lo)/2;
        }
        const uint32_t c   = count_ge(mid);
        if (c >= kk) {
            lo   = mid;
            c_lo = c;
        } else {
            hi = mid;
        }
    }

    if (c_lo > max_gath) {
        // lo is the top_k-th largest key and more than max_gath elements are >= lo
        if (tid == 0) {
            sel[ir] = { 1, 0, 0 };
        }
        return;
    }

    // gather the elements >= lo
    if (tid == 0) {
        s_ng = 0;
    }
    __syncthreads();
    for_each([&](int i, uint32_t key) {
        if (key >= lo) {
            const int j = atomicAdd(&s_ng, 1);
            s_gidx[j] = i;
            s_gkey[j] = key;
        }
    }, idx{});
    __syncthreads();

    const int ng = s_ng;

    // sort by (value desc, index asc): one warp per element computes its rank
    for (int j = warp; j < ng; j += nwarps) {
        const uint32_t kj = s_gkey[j];
        const int      ij = s_gidx[j];
        uint32_t rank = 0;
        for (int l = lane; l < ng; l += WARP_SIZE) {
            rank += s_gkey[l] > kj || (s_gkey[l] == kj && s_gidx[l] < ij);
        }
        rank = warp_reduce(rank, false);
        if (lane == 0) {
            s_sorted[rank] = j;
        }
    }
    __syncthreads();

    // candidates: all elements >= the top_k-th largest
    const uint32_t tau = s_gkey[s_sorted[kk - 1]];
    int nc = kk;
    while (nc < ng && s_gkey[s_sorted[nc]] == tau) {
        ++nc;
    }

    // softmax terms of the sorted candidates (exp in double: the same value as on the CPU)
    if (top_p < 1.0f) {
        for (int l = tid; l < nc; l += nthreads) {
            const float e = (float) exp((double) x[s_gidx[s_sorted[l]]] - (double) x[s_gidx[s_sorted[0]]]);
            s_exp[l] = e < FLT_MIN ? 0.0f : e;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int c = nc;
        if (top_p < 1.0f) {
            float sum = 0.0f;
            for (int l = 0; l < nc; ++l) {
                sum += s_exp[l];
            }
            const float lim = top_p*sum;
            float cum = 0.0f;
            for (int l = 0; l < nc; ++l) {
                cum += s_exp[l];
                if (cum >= lim) {
                    c = l + 1;
                    break;
                }
            }
        }
        const int jl = s_sorted[c - 1];
        sel[ir] = { 0, s_gkey[jl], s_gidx[jl] };
    }
}

template <typename T_ids>
static __global__ void k_gumbel_apply(
        const float * __restrict__ a, const T_ids * __restrict__ ids, const int32_t * __restrict__ par, const gumbel_sel * __restrict__ sel,
        float * __restrict__ dst, const int64_t n, const int64_t s01, const int32_t k_off) {
    const int64_t i  = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t ir = blockIdx.y;

    if (i >= n) {
        return;
    }

    const float t = __int_as_float(par[2]);
    const float x = a[ir*s01 + i];

    if (t == 0.0f) {
        dst[ir*n + i] = x;
        return;
    }

    const int64_t    nv = par[3] > 0 ? min((int64_t) par[3], n) : n;
    const gumbel_sel s  = sel[ir];

    bool keep = i < nv;
    if (keep && !s.all) {
        const uint32_t key = gumbel_key(x);
        keep = key > s.lkey || (key == s.lkey && i <= s.lidx);
    }

    const uint32_t k = (uint32_t) par[1] + (uint32_t) k_off + (uint32_t) ir;

    dst[ir*n + i] = keep ? fmaf(t, gumbel_noise((uint32_t) par[0], k, (int32_t) ids[i]), x) : -INFINITY;
}

void ggml_cuda_op_add_gumbel(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32);
    GGML_ASSERT(src0->nb[0] == sizeof(float) && ggml_is_contiguous(dst) && ggml_is_contiguous(src1));

    const int64_t n   = src0->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t s01 = src0->nb[1]/sizeof(float);

    const int32_t k_off = ggml_get_op_params_i32(dst, 0);

    cudaStream_t stream = ctx.stream();

    const float   * a_d   = (const float   *) src0->data;
    const int32_t * par_d = (const int32_t *) src2->data;
    float         * dst_d = (float         *) dst->data;

    ggml_cuda_pool_alloc<gumbel_sel> sel(ctx.pool(), nr);

    constexpr int nthreads = 1024;
    if (n <= 8*nthreads) {
        k_gumbel_select<nthreads,  8><<<nr, nthreads, 0, stream>>>(a_d, par_d, sel.get(), n, s01);
    } else if (n <= 24*nthreads) {
        k_gumbel_select<nthreads, 24><<<nr, nthreads, 0, stream>>>(a_d, par_d, sel.get(), n, s01);
    } else {
        k_gumbel_select<nthreads,  0><<<nr, nthreads, 0, stream>>>(a_d, par_d, sel.get(), n, s01);
    }

    const int block = 256;
    const dim3 grid((n + block - 1)/block, nr, 1);

    if (src1->type == GGML_TYPE_I32) {
        k_gumbel_apply<<<grid, block, 0, stream>>>(a_d, (const int32_t *) src1->data, par_d, sel.get(), dst_d, n, s01, k_off);
    } else {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        k_gumbel_apply<<<grid, block, 0, stream>>>(a_d, (const float *) src1->data, par_d, sel.get(), dst_d, n, s01, k_off);
    }
}

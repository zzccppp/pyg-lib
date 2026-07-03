#pragma once

#include <ATen/ATen.h>
#include <optional>
#include <tuple>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Fused sparse-dense aggregation (SpMM) in CSR form -- the core neighbor
// aggregation of a GNN message-passing layer, in a single pass.
//
// For a graph in CSR-by-target layout, node `i` owns the edge range
// `[indptr[i], indptr[i+1])`; edge `e` in that range reads source row
// `col[e]` of `x`. The output is
//
//   out[i, f] = REDUCE_{e in row i}  weight[e] * x[col[e], f]
//
// where `reduce` is 0=sum, 1=mean, 2=max, and `weight` (optional, shape `[E]`)
// applies a per-edge scalar (e.g. GCN normalization or attention coefficients);
// when absent it is 1. `mean` divides by the row degree; `max` reduces
// `weight[e] * x[col[e], f]`. Empty rows (`indptr[i+1] == indptr[i]`) produce 0.
//
// Unlike the gather + `scatter_add` path PyG uses for aggregation, this never
// materializes the `[E, F]` message tensor and needs no atomics: each output
// row is owned by one thread that scans its edges sequentially. On MPS this is
// ~9x faster than native `scatter_add` and turns GNN aggregation from a
// CPU-favored bottleneck into an on-device win.
//
// `x` is `[N_src, F]` (2-D), `indptr` is `[N_dst + 1]`, `col` is `[E]` (long).
// `x` may be float32/float16/bfloat16.
PYG_API at::Tensor spmm_csr(const at::Tensor& x,
                            const at::Tensor& indptr,
                            const at::Tensor& col,
                            const std::optional<at::Tensor>& weight,
                            const int64_t reduce);

// Max-reducing SpMM that also returns, per output cell `(i, f)`, the source node
// `col[e*]` of the winning edge (first occurrence on ties) so the backward can
// route the gradient to exactly that neighbor -- the SpMM analogue of
// `segment_max_csr`'s `arg_out`. Empty rows produce value 0 and arg `x.size(0)`
// (one past the last valid source row). `weight` scales the compared value
// `weight[e] * x[col[e], f]`; `arg` still reports the source node id.
//
// Returns `(out, arg)` with `out` shaped `[N_dst, F]` (x's dtype) and `arg`
// `[N_dst, F]` (int64). `arg` is non-differentiable.
PYG_API std::tuple<at::Tensor, at::Tensor> spmm_max_csr(
    const at::Tensor& x,
    const at::Tensor& indptr,
    const at::Tensor& col,
    const std::optional<at::Tensor>& weight);

}  // namespace ops
}  // namespace pyg

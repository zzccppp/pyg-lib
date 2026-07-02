#pragma once

#include <ATen/ATen.h>
#include <optional>
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

}  // namespace ops
}  // namespace pyg

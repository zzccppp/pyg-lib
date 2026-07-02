#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <limits>

namespace pyg {
namespace ops {

namespace {

// out[i, f] = REDUCE_{e in [indptr[i], indptr[i+1])} weight[e] * x[col[e], f].
// reduce: 0=sum, 1=mean, 2=max. Empty rows produce 0.
at::Tensor spmm_csr_kernel(const at::Tensor& x,
                           const at::Tensor& indptr,
                           const at::Tensor& col,
                           const std::optional<at::Tensor>& weight,
                           const int64_t reduce) {
  const auto x_c = x.contiguous();
  const auto indptr_c = indptr.contiguous();
  const auto col_c = col.contiguous();
  const auto N = indptr_c.size(0) - 1;
  const auto F = x_c.size(1);
  auto out = at::zeros({N, F}, x_c.options());

  const bool has_w = weight.has_value();
  const auto w_c = has_w ? weight.value().contiguous() : at::Tensor();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x_c.scalar_type(), "spmm_csr_kernel", [&] {
        const auto* ip = indptr_c.data_ptr<int64_t>();
        const auto* cl = col_c.data_ptr<int64_t>();
        const auto* xp = x_c.data_ptr<scalar_t>();
        auto* op = out.data_ptr<scalar_t>();
        const scalar_t* wp = has_w ? w_c.data_ptr<scalar_t>() : nullptr;

        at::parallel_for(0, N, 1, [&](int64_t beg, int64_t end) {
          for (int64_t i = beg; i < end; ++i) {
            const int64_t start = ip[i], stop = ip[i + 1];
            if (stop <= start)
              continue;
            auto* out_row = op + i * F;
            if (reduce == 2) {  // max
              for (int64_t f = 0; f < F; ++f)
                out_row[f] = std::numeric_limits<scalar_t>::lowest();
            }
            for (int64_t e = start; e < stop; ++e) {
              const auto* x_row = xp + cl[e] * F;
              const scalar_t w = has_w ? wp[e] : scalar_t(1);
              if (reduce == 2) {
                for (int64_t f = 0; f < F; ++f)
                  out_row[f] = std::max(out_row[f], static_cast<scalar_t>(w * x_row[f]));
              } else {
                for (int64_t f = 0; f < F; ++f)
                  out_row[f] += w * x_row[f];
              }
            }
            if (reduce == 1) {  // mean
              const scalar_t deg = static_cast<scalar_t>(stop - start);
              for (int64_t f = 0; f < F; ++f)
                out_row[f] /= deg;
            }
          }
        });
      });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_csr"), TORCH_FN(spmm_csr_kernel));
}

}  // namespace ops
}  // namespace pyg

#include "spmm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

// Fused CSR SpMM aggregation. See spmm.h for the full contract.
PYG_API at::Tensor spmm_csr(const at::Tensor& x,
                            const at::Tensor& indptr,
                            const at::Tensor& col,
                            const std::optional<at::Tensor>& weight,
                            const int64_t reduce) {
  at::TensorArg x_arg{x, "x", 0};
  at::TensorArg indptr_arg{indptr, "indptr", 1};
  at::TensorArg col_arg{col, "col", 2};
  at::CheckedFrom c{"spmm_csr"};

  at::checkAllDefined(c, {x_arg, indptr_arg, col_arg});
  at::checkDim(c, x_arg, 2);
  at::checkDim(c, indptr_arg, 1);
  at::checkDim(c, col_arg, 1);
  TORCH_CHECK(reduce >= 0 && reduce <= 2,
              "spmm_csr: reduce must be 0=sum, 1=mean, 2=max (got ", reduce, ")");
  TORCH_CHECK(x.device() == indptr.device() && x.device() == col.device(),
              "spmm_csr: x, indptr and col must be on the same device");
  if (weight.has_value()) {
    at::TensorArg w_arg{weight.value(), "weight", 3};
    at::checkDim(c, w_arg, 1);
    at::checkSize(c, w_arg, 0, col.size(0));
    TORCH_CHECK(x.device() == weight.value().device(),
                "spmm_csr: x and weight must be on the same device");
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::spmm_csr", "")
                       .typed<decltype(spmm_csr)>();
  return op.call(x, indptr, col, weight, reduce);
}

// Max-reducing SpMM returning (out, arg). See spmm.h for the full contract.
PYG_API std::tuple<at::Tensor, at::Tensor> spmm_max_csr(
    const at::Tensor& x,
    const at::Tensor& indptr,
    const at::Tensor& col,
    const std::optional<at::Tensor>& weight) {
  at::TensorArg x_arg{x, "x", 0};
  at::TensorArg indptr_arg{indptr, "indptr", 1};
  at::TensorArg col_arg{col, "col", 2};
  at::CheckedFrom c{"spmm_max_csr"};

  at::checkAllDefined(c, {x_arg, indptr_arg, col_arg});
  at::checkDim(c, x_arg, 2);
  at::checkDim(c, indptr_arg, 1);
  at::checkDim(c, col_arg, 1);
  TORCH_CHECK(x.device() == indptr.device() && x.device() == col.device(),
              "spmm_max_csr: x, indptr and col must be on the same device");
  if (weight.has_value()) {
    at::TensorArg w_arg{weight.value(), "weight", 3};
    at::checkDim(c, w_arg, 1);
    at::checkSize(c, w_arg, 0, col.size(0));
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::spmm_max_csr", "")
                       .typed<decltype(spmm_max_csr)>();
  return op.call(x, indptr, col, weight);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::spmm_csr(Tensor x, Tensor indptr, Tensor col, Tensor? weight, "
      "int reduce) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "pyg::spmm_max_csr(Tensor x, Tensor indptr, Tensor col, Tensor? weight) "
      "-> (Tensor, Tensor)"));
}

}  // namespace ops
}  // namespace pyg

#include "../spline.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <tuple>

namespace pyg {
namespace ops {
namespace {

std::tuple<at::Tensor, at::Tensor> spline_basis_mps(
    const at::Tensor& pseudo,
    const at::Tensor& kernel_size,
    const at::Tensor& is_open_spline,
    int64_t degree) {
  auto result = spline_basis(pseudo.cpu(), kernel_size.cpu(),
                             is_open_spline.cpu(), degree);
  return std::make_tuple(std::get<0>(result).to(pseudo.device()),
                         std::get<1>(result).to(pseudo.device()));
}

at::Tensor spline_basis_backward_mps(const at::Tensor& grad_basis,
                                     const at::Tensor& pseudo,
                                     const at::Tensor& kernel_size,
                                     const at::Tensor& is_open_spline,
                                     int64_t degree) {
  auto out = spline_basis_backward(grad_basis.cpu(), pseudo.cpu(),
                                   kernel_size.cpu(), is_open_spline.cpu(),
                                   degree);
  return out.to(pseudo.device());
}

at::Tensor spline_weighting_mps(const at::Tensor& x,
                                const at::Tensor& weight,
                                const at::Tensor& basis,
                                const at::Tensor& weight_index) {
  auto out = spline_weighting(x.cpu(), weight.cpu(), basis.cpu(),
                              weight_index.cpu());
  return out.to(x.device());
}

at::Tensor spline_weighting_backward_x_mps(const at::Tensor& grad_out,
                                           const at::Tensor& weight,
                                           const at::Tensor& basis,
                                           const at::Tensor& weight_index) {
  auto out = spline_weighting_backward_x(
      grad_out.cpu(), weight.cpu(), basis.cpu(), weight_index.cpu());
  return out.to(grad_out.device());
}

at::Tensor spline_weighting_backward_weight_mps(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& basis,
    const at::Tensor& weight_index,
    int64_t kernel_size) {
  auto out = spline_weighting_backward_weight(
      grad_out.cpu(), x.cpu(), basis.cpu(), weight_index.cpu(), kernel_size);
  return out.to(grad_out.device());
}

at::Tensor spline_weighting_backward_basis_mps(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& weight_index) {
  auto out = spline_weighting_backward_basis(
      grad_out.cpu(), x.cpu(), weight.cpu(), weight_index.cpu());
  return out.to(grad_out.device());
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, MPS, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_basis"),
         TORCH_FN(spline_basis_mps));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_basis_backward"),
         TORCH_FN(spline_basis_backward_mps));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting"),
         TORCH_FN(spline_weighting_mps));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting_backward_x"),
         TORCH_FN(spline_weighting_backward_x_mps));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting_backward_weight"),
         TORCH_FN(spline_weighting_backward_weight_mps));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting_backward_basis"),
         TORCH_FN(spline_weighting_backward_basis_mps));
}

}  // namespace ops
}  // namespace pyg

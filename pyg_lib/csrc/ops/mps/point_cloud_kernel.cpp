#include "../cluster.h"
#include "../fps.h"
#include "../knn.h"
#include "../nearest.h"
#include "../radius.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace pyg {
namespace ops {
namespace {

std::optional<at::Tensor> cpu_optional(
    const std::optional<at::Tensor>& tensor) {
  if (!tensor.has_value()) {
    return std::nullopt;
  }
  return tensor.value().cpu().contiguous();
}

at::Tensor knn_mps(const at::Tensor& x,
                   const at::Tensor& y,
                   const std::optional<at::Tensor>& ptr_x,
                   const std::optional<at::Tensor>& ptr_y,
                   int64_t k,
                   bool cosine,
                   int64_t num_workers) {
  auto out = knn(x.cpu(), y.cpu(), cpu_optional(ptr_x), cpu_optional(ptr_y), k,
                 cosine, num_workers);
  return out.to(x.device());
}

at::Tensor radius_mps(const at::Tensor& x,
                      const at::Tensor& y,
                      const std::optional<at::Tensor>& ptr_x,
                      const std::optional<at::Tensor>& ptr_y,
                      double r,
                      int64_t max_num_neighbors,
                      int64_t num_workers,
                      bool ignore_same_index) {
  auto out = radius(x.cpu(), y.cpu(), cpu_optional(ptr_x), cpu_optional(ptr_y),
                    r, max_num_neighbors, num_workers, ignore_same_index);
  return out.to(x.device());
}

at::Tensor nearest_mps(const at::Tensor& x,
                       const at::Tensor& y,
                       const std::optional<at::Tensor>& ptr_x,
                       const std::optional<at::Tensor>& ptr_y) {
  auto out =
      nearest(x.cpu(), y.cpu(), cpu_optional(ptr_x), cpu_optional(ptr_y));
  return out.to(x.device());
}

at::Tensor fps_mps(const at::Tensor& src,
                   const at::Tensor& ptr,
                   double ratio,
                   bool random_start) {
  auto out = fps(src.cpu(), ptr.cpu(), ratio, random_start);
  return out.to(src.device());
}

at::Tensor grid_cluster_mps(const at::Tensor& pos,
                            const at::Tensor& size,
                            const std::optional<at::Tensor>& start,
                            const std::optional<at::Tensor>& end) {
  auto out = grid_cluster(pos.cpu(), size.cpu(), cpu_optional(start),
                          cpu_optional(end));
  return out.to(pos.device());
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, MPS, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::knn"), TORCH_FN(knn_mps));
  m.impl(TORCH_SELECTIVE_NAME("pyg::radius"), TORCH_FN(radius_mps));
  m.impl(TORCH_SELECTIVE_NAME("pyg::nearest"), TORCH_FN(nearest_mps));
  m.impl(TORCH_SELECTIVE_NAME("pyg::fps"), TORCH_FN(fps_mps));
  m.impl(TORCH_SELECTIVE_NAME("pyg::grid_cluster"),
         TORCH_FN(grid_cluster_mps));
}

}  // namespace ops
}  // namespace pyg

// Native MPS support for `spmm_csr` -- the fused GNN neighbor-aggregation kernel.
//
//   out[i, f] = REDUCE_{e in [indptr[i], indptr[i+1])} weight[e] * x[col[e], f]
//
// One thread owns each output cell `(i, f)`: it scans node `i`'s edge range,
// reads source row `col[e]` of `x`, applies the optional per-edge `weight`, and
// reduces (sum/mean/max) -- all in a single pass, with no `[E, F]` message
// tensor and no atomics (each output row has a unique owner). This replaces the
// gather + `scatter_add` path PyG uses for aggregation, whose MPS `scatter_add`
// is pathologically slow under the index collisions every GNN produces.
//
// Hot path: 2-D `x` (f32/f16/bf16), 1-D int64 `indptr`/`col`, `weight` (if any)
// matching `x`'s dtype, output fitting 32-bit counters. Anything else falls back
// to the CPU kernel.

#include "../spmm.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/mps.h>

#include <limits>

namespace pyg {
namespace ops {
namespace {

static const char* kShaderSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

inline float to_float(float v)  { return v; }
inline float to_float(half v)   { return float(v); }
inline float to_float(ushort v) { return as_type<float>(uint(v) << 16); }
inline void store_out(device float* p, uint i, float v)  { p[i] = v; }
inline void store_out(device half* p, uint i, float v)   { p[i] = half(v); }
inline void store_out(device ushort* p, uint i, float v) { p[i] = ushort(as_type<uint>(v) >> 16); }

// reduce: 0=sum, 1=mean, 2=max
template <typename T>
inline void spmm_impl(device const T* x, device const long* indptr,
                      device const long* col, device const T* weight,
                      device T* out, uint N, uint F, uint has_w, uint reduce,
                      uint gid) {
    if (gid >= N * F) return;
    uint i = gid / F;
    uint f = gid % F;
    long start = indptr[i], end = indptr[i + 1];
    if (end <= start) { store_out(out, gid, 0.0f); return; }
    float acc = (reduce == 2u) ? -INFINITY : 0.0f;
    for (long e = start; e < end; ++e) {
        float w = (has_w != 0u) ? to_float(weight[e]) : 1.0f;
        float v = w * to_float(x[(uint)col[e] * F + f]);
        acc = (reduce == 2u) ? max(acc, v) : (acc + v);
    }
    if (reduce == 1u) acc /= float(end - start);
    store_out(out, gid, acc);
}

#define SPMM_K(NAME, T)                                       \
  kernel void NAME(device const T* x [[buffer(0)]],           \
                   device const long* indptr [[buffer(1)]],   \
                   device const long* col [[buffer(2)]],      \
                   device const T* weight [[buffer(3)]],      \
                   device T* out [[buffer(4)]],               \
                   constant uint& N [[buffer(5)]],            \
                   constant uint& F [[buffer(6)]],            \
                   constant uint& has_w [[buffer(7)]],        \
                   constant uint& reduce [[buffer(8)]],       \
                   uint gid [[thread_position_in_grid]]) {    \
    spmm_impl(x, indptr, col, weight, out, N, F, has_w, reduce, gid); \
  }

SPMM_K(spmm_f32, float)
SPMM_K(spmm_f16, half)
SPMM_K(spmm_bf16, ushort)
)METAL";

constexpr int kNumDtypes = 3;

struct MetalSpmm {
  id<MTLComputePipelineState> pso[kNumDtypes] = {nil, nil, nil};
  bool ok = false;
};

id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib,
                                     NSString* name) {
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  if (!fn) return nil;
  NSError* err = nil;
  return [dev newComputePipelineStateWithFunction:fn error:&err];
}

const MetalSpmm& metal_spmm() {
  static MetalSpmm state = [] {
    MetalSpmm s;
    @autoreleasepool {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      if (!dev) return s;
      NSError* err = nil;
      id<MTLLibrary> lib = [dev
          newLibraryWithSource:[NSString stringWithUTF8String:kShaderSrc]
                       options:nil
                         error:&err];
      if (!lib) return s;
      NSString* names[kNumDtypes] = {@"spmm_f32", @"spmm_f16", @"spmm_bf16"};
      for (int i = 0; i < kNumDtypes; ++i) {
        s.pso[i] = make_pso(dev, lib, names[i]);
        if (!s.pso[i]) return s;
      }
      s.ok = true;
    }
    return s;
  }();
  return state;
}

int dtype_index(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return 0;
    case at::kHalf: return 1;
    case at::kBFloat16: return 2;
    default: return -1;
  }
}

static inline id<MTLBuffer> mtl_buffer(const at::Tensor& t) {
  return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}
static inline NSUInteger byte_offset(const at::Tensor& t) {
  return (NSUInteger)(t.storage_offset() * t.element_size());
}

bool spmm_eligible(const at::Tensor& x, const at::Tensor& indptr,
                   const at::Tensor& col,
                   const std::optional<at::Tensor>& weight, int64_t N) {
  const int64_t u32 = std::numeric_limits<uint32_t>::max();
  const bool w_ok = !weight.has_value() ||
                    (weight->dim() == 1 &&
                     weight->scalar_type() == x.scalar_type());
  return metal_spmm().ok && x.dim() == 2 && dtype_index(x.scalar_type()) >= 0 &&
         indptr.scalar_type() == at::kLong && col.scalar_type() == at::kLong &&
         w_ok && N * x.size(1) <= u32;
}

at::Tensor spmm_csr_cpu_fallback(const at::Tensor& x, const at::Tensor& indptr,
                                 const at::Tensor& col,
                                 const std::optional<at::Tensor>& weight,
                                 int64_t reduce) {
  auto w = weight.has_value()
               ? std::optional<at::Tensor>(weight->cpu())
               : std::nullopt;
  return spmm_csr(x.cpu(), indptr.cpu(), col.cpu(), w, reduce).to(x.device());
}

at::Tensor spmm_csr_mps(const at::Tensor& x, const at::Tensor& indptr,
                        const at::Tensor& col,
                        const std::optional<at::Tensor>& weight,
                        const int64_t reduce) {
  const int64_t N = indptr.size(0) - 1;
  if (!spmm_eligible(x, indptr, col, weight, N)) {
    return spmm_csr_cpu_fallback(x, indptr, col, weight, reduce);
  }
  auto x_c = x.contiguous();
  auto indptr_c = indptr.contiguous();
  auto col_c = col.contiguous();
  const uint32_t Nu = (uint32_t)N;
  const uint32_t F = (uint32_t)x_c.size(1);
  const uint32_t has_w = weight.has_value() ? 1u : 0u;
  const uint32_t red = (uint32_t)reduce;
  auto w_c = weight.has_value() ? weight->contiguous() : x_c;  // filler if absent

  auto out = at::empty({N, (int64_t)F}, x_c.options());
  const uint32_t total = Nu * F;
  auto pso = metal_spmm().pso[dtype_index(x_c.scalar_type())];
  @autoreleasepool {
    id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:mtl_buffer(x_c) offset:byte_offset(x_c) atIndex:0];
      [enc setBuffer:mtl_buffer(indptr_c) offset:byte_offset(indptr_c) atIndex:1];
      [enc setBuffer:mtl_buffer(col_c) offset:byte_offset(col_c) atIndex:2];
      [enc setBuffer:mtl_buffer(w_c) offset:byte_offset(w_c) atIndex:3];
      [enc setBuffer:mtl_buffer(out) offset:0 atIndex:4];
      [enc setBytes:&Nu length:sizeof(Nu) atIndex:5];
      [enc setBytes:&F length:sizeof(F) atIndex:6];
      [enc setBytes:&has_w length:sizeof(has_w) atIndex:7];
      [enc setBytes:&red length:sizeof(red) atIndex:8];
      NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup,
                          (NSUInteger)total);
      [enc dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(MAX(tg, (NSUInteger)1), 1, 1)];
      [enc endEncoding];
      torch::mps::commit();
    });
  }
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, MPS, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_csr"), TORCH_FN(spmm_csr_mps));
}

}  // namespace ops
}  // namespace pyg

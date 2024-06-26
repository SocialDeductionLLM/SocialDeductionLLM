#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

void cuda_forward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v,
                  float *last_state, bf16 *y, float *new_state);
void cuda_backward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v,
                   float *last_state, bf16 *gy, float *gnew_state, bf16 *gw,
                   bf16 *gu, bf16 *gk, bf16 *gv, float *glast_state);

using namespace torch::autograd;

class WKVFunction : public Function<WKVFunction> {
public:
  static tensor_list forward(AutogradContext *ctx, torch::Tensor raw_w,
                             torch::Tensor raw_u, torch::Tensor raw_k,
                             torch::Tensor raw_v,
                             torch::Tensor raw_last_state) {
    // Input sizes sanity check
    assert(raw_k.dim() == 3);
    const auto [B, T, C] =
        std::tuple(raw_k.size(0), raw_k.size(1), raw_k.size(2));
    assert(T <= Tmax);
    assert(raw_w.dim() == 1 && raw_w.size(0) == C);
    assert(raw_u.dim() == 1 && raw_u.size(0) == C);
    assert(raw_v.dim() == 3 && raw_v.size(0) == B && raw_v.size(1) == T &&
           raw_v.size(2) == C);
    assert(raw_last_state.dim() == 3 && raw_last_state.size(0) == B &&
           raw_last_state.size(1) == C && raw_last_state.size(2) == 3);

    // Enforce alignment for CUDA kernel
    assert(B * C % std::min(C, 32l) == 0);

    const auto w = -raw_w.toType(c10::ScalarType::Float).contiguous().exp();
    const auto u = raw_u.contiguous();
    const auto k = raw_k.contiguous();
    const auto v = raw_v.contiguous();
    const auto last_state = raw_last_state.contiguous();
    auto y = torch::empty_like(k);
    auto new_state = torch::empty_like(last_state);

    assert(y.is_contiguous());
    assert(new_state.is_contiguous());

    assert(w.dtype().isScalarType(torch::ScalarType::Float));
    assert(u.dtype().isScalarType(torch::ScalarType::BFloat16));
    assert(k.dtype().isScalarType(torch::ScalarType::BFloat16));
    assert(v.dtype().isScalarType(torch::ScalarType::BFloat16));
    assert(last_state.dtype().isScalarType(torch::ScalarType::Float));
    assert(y.dtype().isScalarType(torch::ScalarType::BFloat16));
    assert(new_state.dtype().isScalarType(torch::ScalarType::Float));

    cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(),
                 k.data_ptr<bf16>(), v.data_ptr<bf16>(),
                 last_state.data_ptr<float>(), y.data_ptr<bf16>(),
                 new_state.data_ptr<float>());

    ctx->save_for_backward({w, u, k, v, last_state});

    return {y, new_state};
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto gy = grad_outputs[0], gnew_state = grad_outputs[1];
    auto inputs = ctx->get_saved_variables();
    auto w = inputs[0], u = inputs[1], k = inputs[2], v = inputs[3],
         last_state = inputs[4];
    const auto [B, T, C] = std::tuple(k.size(0), k.size(1), k.size(2));

    auto gw =
        torch::empty({B, C}, u.options(), torch::MemoryFormat::Contiguous);
    auto gu =
        torch::empty({B, C}, u.options(), torch::MemoryFormat::Contiguous);
    auto gk = torch::empty_like(k);
    auto gv = torch::empty_like(v);
    auto glast_state = torch::empty_like(last_state);

    cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(),
                  k.data_ptr<bf16>(), v.data_ptr<bf16>(),
                  last_state.data_ptr<float>(), gy.data_ptr<bf16>(),
                  gnew_state.data_ptr<float>(), gw.data_ptr<bf16>(),
                  gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>(),
                  glast_state.data_ptr<float>());

    return {gw.sum({0}), gu.sum({0}), gk, gv, glast_state};
  }
};

tensor_list wkv(torch::Tensor w, torch::Tensor u, torch::Tensor k,
                torch::Tensor v, torch::Tensor last_state) {
  return WKVFunction::apply(w, u, k, v, last_state);
}



//void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y);
//void cuda_forward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y);
//void cuda_forward_with_state(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *s, long *lengths);
void cuda_forward_with_state_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, float *s, long *lengths);
//void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *gy, float *gw, float *gu, float *gk, float *gv);
//void cuda_backward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, bf16 *gy, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv);

// void forward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
//     const int B = k.size(0);
//     const int T = k.size(1);
//     const int C = k.size(2);
//     cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>());
// }
// void forward_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
//     const int B = k.size(0);
//     const int T = k.size(1);
//     const int C = k.size(2);
//     cuda_forward_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>());
// }
// void forward_with_state(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s, torch::Tensor &lengths) {
//     const int B = k.size(0);
//     const int T = k.size(1);
//     const int C = k.size(2);
//     cuda_forward_with_state(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), s.data_ptr<float>(), lengths.data_ptr<long>());
// }
void forward_with_state_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s, torch::Tensor &lengths) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    cuda_forward_with_state_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(), s.data_ptr<float>(), lengths.data_ptr<long>());
}
// void backward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
//     const int B = k.size(0);
//     const int T = k.size(1);
//     const int C = k.size(2);
//     cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), gy.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>());
// }
// void backward_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
//     const int B = k.size(0);
//     const int T = k.size(1);
//     const int C = k.size(2);
//     cuda_backward_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(),
//         gy.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>());
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("forward", &forward, "wkv forward");
    // m.def("forward_bf16", &forward_bf16, "wkv forward bf16");
    // m.def("forward_with_state", &forward_with_state, "wkv forward with state");
    m.def("forward_with_state_bf16", &forward_with_state_bf16, "wkv forward with state bf16");
    // m.def("backward", &backward, "wkv backward");
    // m.def("backward_bf16", &backward_bf16, "wkv backward bf16");
    m.def("wkv", &wkv, "WKV operator for RWKV");
}

TORCH_LIBRARY(rwkv, m) {
    // m.def("forward", forward);
    // m.def("forward_bf16", forward_bf16);
    // m.def("forward_with_state", forward_with_state);
    m.def("forward_with_state_bf16", forward_with_state_bf16);
    // m.def("backward", backward);
    // m.def("backward_bf16", backward_bf16);
    m.def("wkv", &wkv);
}

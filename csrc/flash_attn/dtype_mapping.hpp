#include <torch/torch.h>
#include <cutlass/numeric_types.h>


namespace test_fa{

template <typename T>
struct CutlassToTorch;

// Note torch::ScalarType is not a type, is an enum, so we cannot use type aliasing
template <>
struct CutlassToTorch<cutlass::half_t> {
  static constexpr torch::ScalarType value = torch::kFloat16;
};

template <>
struct CutlassToTorch<cutlass::bfloat16_t> {
  static constexpr torch::ScalarType value = torch::kBFloat16;
};

template <>
struct CutlassToTorch<cutlass::tfloat32_t> {
  static constexpr torch::ScalarType value = torch::kFloat32;
};

template <torch::ScalarType>
struct TorchToCutlass;

template <>
struct TorchToCutlass<torch::kFloat16> {
    using type = cutlass::half_t;
};

template <>
struct TorchToCutlass<torch::kBFloat16> {
    using type = cutlass::bfloat16_t;
};

template <>
struct TorchToCutlass<torch::kFloat32> {
    using type = cutlass::tfloat32_t;
};

}

#include "operators/matrix_size.hpp"
#include <cmath>

size_t precision_element_size(Precision p) {
  switch (p) {
    case Precision::FP64: return sizeof(double);
    case Precision::FP32:
    case Precision::TF32: return sizeof(float);
    case Precision::FP16:
    case Precision::BF16: return 2;
    case Precision::INT8: return 1;
    case Precision::INT4: return 0.5; // 2 elements per byte
    case Precision::FP8:  return 1;
    case Precision::FP4:  return 0.5; // 2 elements per byte
  }
  return 1;
}

std::tuple<int, int, int> adaptive_matrix_size(
    const OperatorDescriptor& desc,
    size_t total_vram,
    double target_mem_fraction) {
    // GEMM: C = A @ B, A is m×k, B is k×n, C is m×n
    // Memory = (m*k + k*n + m*n) * element_size
    // For square matrices m=n=k, memory = 3*k*k*element_size
    size_t elem_size = precision_element_size(desc.precision);
    size_t usable_mem = static_cast<size_t>(total_vram * target_mem_fraction);
    size_t mem_budget = usable_mem / elem_size;

    int k = static_cast<int>(std::sqrt(mem_budget / 3.0));
    k = std::max(k, 256);
    int base = (k / 256) * 256;
    if (base < 256) base = 256;

    return {base, base, base};
}

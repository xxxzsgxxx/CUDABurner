#include "operators/matrix_size.hpp"
#include <cmath>

size_t precision_element_size(Precision p) {
  switch (p) {
    case Precision::FP64: return 8;
    case Precision::FP32:
    case Precision::TF32: return 4;
    case Precision::FP16:
    case Precision::BF16: return 2;
    case Precision::INT8: case Precision::FP8: return 1;
    case Precision::INT4: return 1; // sub-byte (2 elems/byte), can't represent in size_t
    case Precision::FP4:  return 1;
    default: return 1;
  }
}

std::tuple<int, int, int> adaptive_matrix_size(
    const OperatorDescriptor& desc,
    size_t total_vram,
    double target_mem_fraction,
    int max_dim) {
    double elem_size = 1.0;
    switch (desc.precision) {
        case Precision::FP64: elem_size = 8; break;
        case Precision::FP32: case Precision::TF32: elem_size = 4; break;
        case Precision::FP16: case Precision::BF16: elem_size = 2; break;
        case Precision::INT8: case Precision::FP8: elem_size = 1; break;
        case Precision::INT4: case Precision::FP4: elem_size = 0.5; break;
    }
    size_t usable_mem = static_cast<size_t>(total_vram * target_mem_fraction);
    size_t mem_budget = static_cast<size_t>(usable_mem / elem_size);

    int k = static_cast<int>(std::sqrt(mem_budget / 3.0));
    k = std::max(k, 256);
    k = std::min(k, max_dim);
    int base = (k / 256) * 256;
    if (base < 256) base = 256;

    return {base, base, base};
}

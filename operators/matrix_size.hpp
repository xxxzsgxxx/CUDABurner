#pragma once
#include "operator_traits.hpp"

size_t precision_element_size(Precision p);

std::tuple<int, int, int> adaptive_matrix_size(
    const OperatorDescriptor& desc,
    size_t total_vram,
    double target_mem_fraction = 0.6,
    int max_dim = 8000
);

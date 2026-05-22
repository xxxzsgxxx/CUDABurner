#pragma once
#include "operator_traits.hpp"

size_t precision_element_size(Precision p);

// Returns {m, n, k} adaptive to VRAM and precision
// target_mem_fraction: what fraction of VRAM to use (default 0.6 = 60%)
// For GEMM C = A @ B, total mem = m*k + k*n + m*n elements
std::tuple<int, int, int> adaptive_matrix_size(
    const OperatorDescriptor& desc,
    size_t total_vram,
    double target_mem_fraction = 0.6
);

#include "operators/operator_factory.hpp"
#include "operators/gemm_generic.hpp"
#include "operators/gemm_tensorcore.hpp"
#include "operators/gemm_sparse_tensorcore.hpp"
#include "operators/gemm_fp8_tensorcore.hpp"
#include "operators/gemm_fp4_tensorcore.hpp"
#include "operators/gemm_int4_tensorcore.hpp"
#include "operators/matrix_size.hpp"
#include <tuple>

namespace OperatorFactory {

std::unique_ptr<BaseOperator> create_gemm_operator(
    const OperatorDescriptor& desc, const GpuProperties& props, std::string& out_notes) {

  std::tuple<int, int, int> sizes = adaptive_matrix_size(desc, props.total_global_mem);
  int m, n, k;
  std::tie(m, n, k) = sizes;

  if (desc.sparsity == Sparsity::SPARSE) {
    if (props.cc_major >= 9) {
      if (desc.precision == Precision::FP8) {
        out_notes = "Sparse Tensor Core";
#if CUDA_VERSION >= 11080
        return std::make_unique<GemmFp8TensorCore>(desc, m, n, k);
#else
        out_notes = "FP8 requires CUDA Toolkit 11.8+";
        return nullptr;
#endif
      }
    }
    if (props.cc_major >= 8) {
      if (desc.precision == Precision::FP16 || desc.precision == Precision::INT8) {
        out_notes = "Sparse Tensor Core";
        return std::make_unique<GemmSparseTensorCore>(desc, m, n, k);
      }
    }
  }

  switch (desc.precision) {
    case Precision::FP64:
      if (props.cc_major >= 6) {
        out_notes = "CUDA Core (Native)";
        return std::make_unique<GemmGeneric<double>>(desc, m, n, k, true);
      }
      break;

    case Precision::FP32:
      out_notes = "CUDA Core (TC Disabled)";
      return std::make_unique<GemmGeneric<float>>(desc, m, n, k, true);

    case Precision::TF32:
      if (props.cc_major >= 8) {
        out_notes = "Tensor Core (Default FP32)";
        return std::make_unique<GemmTensorCore>(desc, props, m, n, k);
      }
      break;

    case Precision::FP16:
    case Precision::BF16:
    case Precision::INT8:
      if (props.cc_major >= 7) {
        out_notes = "Tensor Core";
        return std::make_unique<GemmTensorCore>(desc, props, m, n, k);
      }
      break;

    case Precision::INT4:
      {
        bool is_supported = (props.cc_major == 9) || (props.cc_major == 8 && props.cc_minor == 9);
#if CUDA_VERSION < 12030
        is_supported = false;
#endif

        if (is_supported) {
          out_notes = "Tensor Core";
          return std::make_unique<GemmInt4TensorCore>(desc, m, n, k);
        } else {
          out_notes = "INT4 Not Supported On This GPU";
          return nullptr;
        }
      }

    case Precision::FP8:
      {
        bool is_supported = (props.cc_major == 9) || (props.cc_major == 8 && props.cc_minor == 9);
#if CUDA_VERSION < 11080
        is_supported = false;
#endif

        if (is_supported) {
          out_notes = "Tensor Core";
          return std::make_unique<GemmFp8TensorCore>(desc, m, n, k);
        } else {
          out_notes = "FP8 Not Supported On This GPU";
          return nullptr;
        }
      }

    case Precision::FP4:
      {
        bool is_supported = (props.cc_major == 9) || (props.cc_major == 8 && props.cc_minor == 9);
#if CUDA_VERSION < 12030
        is_supported = false;
#endif

        if (is_supported) {
          out_notes = "Tensor Core";
          return std::make_unique<GemmFp4TensorCore>(desc, m, n, k);
        } else {
          out_notes = "FP4 Not Supported On This GPU";
          return nullptr;
        }
      }
  }

  out_notes = "Not Supported";
  return nullptr;
}

}

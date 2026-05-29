#include "operators/operator_factory.hpp"
#include "operators/gemm_generic.hpp"
#include "operators/gemm_tensorcore.hpp"
#include <cublasLt.h>
#include "operators/gemm_sparse_tensorcore.hpp"
#include "operators/gemm_fp8_tensorcore.hpp"
#include "operators/gemm_fp4_tensorcore.hpp"
#include "operators/gemm_int4_tensorcore.hpp"
#include "operators/matrix_size.hpp"
#include <tuple>

namespace OperatorFactory {

// Probe cuBLASLt for algorithm availability — catches cases where CC is sufficient
// but the driver/cuBLAS version doesn't actually support the data type.
static bool probe_cublaslt_gemm_support(
    cudaDataType_t input_type, cudaDataType_t output_type,
    cublasComputeType_t compute_type, int m = 128, int n = 128, int k = 128) {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t A_desc = nullptr, B_desc = nullptr, C_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    bool supported = false;

    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;

    if (cublasLtMatmulDescCreate(&matmul_desc, compute_type, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
        goto out;

    if (cublasLtMatrixLayoutCreate(&A_desc, input_type, m, k, m) != CUBLAS_STATUS_SUCCESS)
        goto out;
    if (cublasLtMatrixLayoutCreate(&B_desc, input_type, k, n, k) != CUBLAS_STATUS_SUCCESS)
        goto out;
    if (cublasLtMatrixLayoutCreate(&C_desc, output_type, m, n, m) != CUBLAS_STATUS_SUCCESS)
        goto out;

    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS)
        goto out;

    cublasLtMatmulHeuristicResult_t heuristic_result;
    int returned_results = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, A_desc, B_desc, C_desc, C_desc,
        preference, 1, &heuristic_result, &returned_results);

    supported = (status == CUBLAS_STATUS_SUCCESS && returned_results > 0);

out:
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (C_desc) cublasLtMatrixLayoutDestroy(C_desc);
    if (B_desc) cublasLtMatrixLayoutDestroy(B_desc);
    if (A_desc) cublasLtMatrixLayoutDestroy(A_desc);
    if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
    if (handle) cublasLtDestroy(handle);
    return supported;
}

std::unique_ptr<BaseOperator> create_gemm_operator(
    const OperatorDescriptor& desc, const GpuProperties& props, std::string& out_notes) {

  if (!props.supports_precision(desc.precision, desc.sparsity)) {
    int sm = props.sm_version();
    switch (desc.precision) {
      case Precision::FP8:
#if CUDA_VERSION < 11080
        out_notes = "FP8 requires CUDA Toolkit 11.8+";
#else
        out_notes = "FP8 requires Ada (RTX 40), Hopper, or Blackwell GPU";
#endif
        return nullptr;
      case Precision::FP4:
#if CUDA_VERSION < 12030
        out_notes = "FP4 requires CUDA Toolkit 12.3+";
#else
        out_notes = "FP4 requires Blackwell GPU (RTX 50 series)";
#endif
        return nullptr;
      case Precision::INT4:
#if CUDA_VERSION < 12030
        out_notes = "INT4 requires CUDA Toolkit 12.3+";
#else
        out_notes = "INT4 requires Blackwell GPU (RTX 50 series)";
#endif
        return nullptr;
      case Precision::INT8:
        if (desc.sparsity == Sparsity::SPARSE) {
          out_notes = "INT8 Sparse requires Ampere+ GPU";
        } else {
          out_notes = "INT8 Tensor Core requires Volta+ GPU";
        }
        return nullptr;
      default:
        out_notes = PRECISION_NAMES.at(desc.precision) + " not supported on this GPU";
        return nullptr;
    }
  }

  std::tuple<int, int, int> sizes = adaptive_matrix_size(desc, props.total_global_mem);
  int m, n, k;
  std::tie(m, n, k) = sizes;

  // --- Sparse operators ---
  if (desc.sparsity == Sparsity::SPARSE) {
    if (desc.precision == Precision::INT8 || desc.precision == Precision::FP16) {
      out_notes = "Sparse Tensor Core";
      try {
        return std::make_unique<GemmSparseTensorCore>(desc, m, n, k);
      } catch (const std::exception& e) {
        out_notes = "Sparse " + PRECISION_NAMES.at(desc.precision) + ": runtime error - " + e.what();
        return nullptr;
      }
    }
    // FP8/FP4/INT4 sparse not implemented
    out_notes = "Sparse " + PRECISION_NAMES.at(desc.precision) + " not yet implemented";
    return nullptr;
  }

  // --- Dense operators ---
  switch (desc.precision) {
    case Precision::FP64:
      if (props.sm_version() >= 60) {
        out_notes = "CUDA Core (Native)";
        return std::make_unique<GemmGeneric<double>>(desc, m, n, k, true);
      }
      break;

    case Precision::FP32:
      out_notes = "CUDA Core (TC Disabled)";
      return std::make_unique<GemmGeneric<float>>(desc, m, n, k, true);

    case Precision::TF32:
      if (props.sm_version() >= 80) {
        out_notes = "Tensor Core (Default FP32)";
        return std::make_unique<GemmTensorCore>(desc, props, m, n, k);
      }
      break;

    case Precision::FP16:
    case Precision::BF16:
    case Precision::INT8:
      if (props.sm_version() >= 70) {
        out_notes = "Tensor Core";
        return std::make_unique<GemmTensorCore>(desc, props, m, n, k);
      }
      break;

    case Precision::INT4:
      if (!probe_cublaslt_gemm_support(CUDA_R_4I, CUDA_R_32I, CUBLAS_COMPUTE_32I)) {
        out_notes = "INT4: cuBLASLt runtime unsupported";
        return nullptr;
      }
      out_notes = "Tensor Core";
      try {
        return std::make_unique<GemmInt4TensorCore>(desc, m, n, k);
      } catch (const std::exception& e) {
        out_notes = "INT4: runtime error - " + std::string(e.what());
        return nullptr;
      }

    case Precision::FP8:
      if (!probe_cublaslt_gemm_support(CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUBLAS_COMPUTE_32F)) {
        out_notes = "FP8: cuBLASLt runtime unsupported";
        return nullptr;
      }
      out_notes = "Tensor Core";
      try {
        return std::make_unique<GemmFp8TensorCore>(desc, m, n, k);
      } catch (const std::exception& e) {
        out_notes = "FP8: runtime error - " + std::string(e.what());
        return nullptr;
      }

    case Precision::FP4:
      if (!probe_cublaslt_gemm_support(CUDA_R_4F_E2M1, CUDA_R_16F, CUBLAS_COMPUTE_32F)) {
        out_notes = "FP4: cuBLASLt runtime unsupported";
        return nullptr;
      }
      out_notes = "Tensor Core";
      try {
        return std::make_unique<GemmFp4TensorCore>(desc, m, n, k);
      } catch (const std::exception& e) {
        out_notes = "FP4: runtime error - " + std::string(e.what());
        return nullptr;
      }
  }

  out_notes = "Not Supported";
  return nullptr;
}

}
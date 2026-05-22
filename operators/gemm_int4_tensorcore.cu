#include "operators/gemm_int4_tensorcore.hpp"
#include "utils/helpers.hpp"

#if CUDA_VERSION >= 12030

#define CUBLASLT_CHECK(call)                                                \
    do {                                                                    \
        cublasStatus_t status = call;                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            std::string errMsg = "cuBLASLt Error in " + std::string(__FILE__) + \
                                 ":" + std::to_string(__LINE__) + " - " +    \
                                 std::to_string(status);                     \
            throw std::runtime_error(errMsg);                               \
        }                                                                   \
    } while (0)

GemmInt4TensorCore::GemmInt4TensorCore(const OperatorDescriptor& desc, int m, int n, int k)
    : descriptor_(desc), m_(m), n_(n), k_(k) {
    
    CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle_));

    cudaDataType_t data_type = CUDA_R_4I;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32I;
    
    // INT4 has 2 elements per byte
    size_t size_A = (size_t)m_ * k_ / 2;
    size_t size_B = (size_t)k_ * n_ / 2;
    size_t size_C = (size_t)m_ * n_;

    CUDA_CHECK(cudaMalloc(&d_A_, size_A));
    CUDA_CHECK(cudaMalloc(&d_B_, size_B));
    CUDA_CHECK(cudaMalloc(&d_C_, size_C));

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmul_desc_, compute_type, CUDA_R_32I));

    size_t lda = m_;
    size_t ldb = k_;
    size_t ldc = m_;

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&A_desc_, data_type, m_, k_, lda));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&B_desc_, data_type, k_, n_, ldb));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&C_desc_, data_type, m_, n_, ldc));

    gflops_or_gops_ = (2.0 * m_ * n_ * k_) / 1e9;
}

GemmInt4TensorCore::~GemmInt4TensorCore() {
    cudaFree(d_A_);
    cudaFree(d_B_);
    cudaFree(d_C_);
    if (A_desc_) cublasLtMatrixLayoutDestroy(A_desc_);
    if (B_desc_) cublasLtMatrixLayoutDestroy(B_desc_);
    if (C_desc_) cublasLtMatrixLayoutDestroy(C_desc_);
    if (matmul_desc_) cublasLtMatmulDescDestroy(matmul_desc_);
    if (cublaslt_handle_) cublasLtDestroy(cublaslt_handle_);
}

void GemmInt4TensorCore::execute(cudaStream_t stream) {
    const int alpha = 1;
    const int beta = 0;
    CUBLASLT_CHECK(cublasLtMatmul(cublaslt_handle_, matmul_desc_, &alpha, d_A_, A_desc_, d_B_, B_desc_, &beta, d_C_, C_desc_, d_C_, C_desc_, NULL, NULL, 0, stream));
}

double GemmInt4TensorCore::get_gflops_or_gops() const {
    return gflops_or_gops_;
}

#endif // CUDA_VERSION >= 12030

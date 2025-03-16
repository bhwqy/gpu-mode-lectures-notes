#include <cuda_runtime.h>
#include <cublas_v2.h>

// modified from: https://zhuanlan.zhihu.com/p/657632577

void cublas_sgemm(float *A, float *B, float *C,  size_t M, size_t N, size_t K) {

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    static float alpha = 1.0;
    static float beta = 0.0;

    cublasGemmEx(handle, 
        CUBLAS_OP_N, 
        CUBLAS_OP_N, 
        N, M, K, 
        &alpha, 
        B, CUDA_R_32F, N, 
        A, CUDA_R_32F, K, 
        &beta,  
        C, CUDA_R_32F, N, 
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
}




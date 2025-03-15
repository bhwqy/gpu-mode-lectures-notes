#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdio>
#include <algorithm>
#include <random>
#include <iostream>
#include <chrono>

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void tensor_core_fp16_gemm(half* a, half* b, float* c, int m, int n, int k) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int inner_k_loop = k / WMMA_K;
    int a_row = WMMA_M * bx;
    int b_col = WMMA_N * by;
    for(int _k = 0; _k < inner_k_loop; ++_k){
        // 2. load fragment A B, 
        nvcuda::wmma::load_matrix_sync(a_frag, &a[a_row * k + _k * WMMA_K], k);
        nvcuda::wmma::load_matrix_sync(b_frag, &b[n * _k * WMMA_K + b_col ], n);

        // 3. tensorcore计算
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4. store C
    nvcuda::wmma::store_matrix_sync(&c[a_row * n + b_col], c_frag, n, nvcuda::wmma::mem_row_major);
}

void float_to_fp16(half *dst, float *src, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2half(src[i]);
    }
}

void fp16_to_float(float *dst, half *src, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __half2float(src[i]);
    }
}

void float_to_fp16_to_float(float* src, float* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __half2float(__float2half(src[i]));
    }
}

void cpu_gemm(float *a, float *b, float *c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// 比较两个矩阵的差异，并统计不同精度范围内的元素数量
void compare_matrices(float *a, float *b, int size) {
    int count_1e_2 = 0;
    int count_1e_3 = 0;
    int count_1e_4 = 0;
    int count_1e_5 = 0;
    int count_1e_6 = 0;

    float max_diff = 0.0f;
    float sum_diff = 0.0f;

    for (int i = 0; i < size; i++) {
        float diff = std::abs(a[i] - b[i]);
        sum_diff += diff;
        max_diff = std::max(max_diff, diff);

        if (diff < 1e-6) count_1e_6++;
        else if (diff < 1e-5) count_1e_5++;
        else if (diff < 1e-4) count_1e_4++;
        else if (diff < 1e-3) count_1e_3++;
        else if (diff < 1e-2) count_1e_2++;
    }

    float avg_diff = sum_diff / size;
    int count_large = size - count_1e_6 - count_1e_5 - count_1e_4 - count_1e_3 - count_1e_2;

    std::cout << "\n----- COMPARISON RESULTS -----\n" << std::endl;
    std::cout << "Maximum Error: " << max_diff << std::endl;
    std::cout << "Average Error: " << avg_diff << std::endl;
    std::cout << "Total Elements: " << size << "\n" << std::endl;

    // Print ASCII table
    std::cout << "+----------------+-------------+---------------+" << std::endl;
    std::cout << "| Error Range    | Count       | Percentage    |" << std::endl;
    std::cout << "+----------------+-------------+---------------+" << std::endl;

    // Print table rows with formatted data
    printf("| < 1e-6         | %-11d | %6.2f%%       |\n", count_1e_6, (float)count_1e_6/size*100);
    printf("| [1e-6, 1e-5)   | %-11d | %6.2f%%       |\n", count_1e_5, (float)count_1e_5/size*100);
    printf("| [1e-5, 1e-4)   | %-11d | %6.2f%%       |\n", count_1e_4, (float)count_1e_4/size*100);
    printf("| [1e-4, 1e-3)   | %-11d | %6.2f%%       |\n", count_1e_3, (float)count_1e_3/size*100);
    printf("| [1e-3, 1e-2)   | %-11d | %6.2f%%       |\n", count_1e_2, (float)count_1e_2/size*100);
    printf("| >= 1e-2        | %-11d | %6.2f%%       |\n", count_large, (float)count_large/size*100);

    std::cout << "+----------------+-------------+---------------+" << std::endl;
}

int main() {
    // 设置随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 分配和初始化主机内存
    float *h_a = new float[M * K];
    float *h_b = new float[K * N];
    float *h_c_gpu = new float[M * N];
    float *h_c_cpu = new float[M * N];

    // 随机初始化输入矩阵
    for (int i = 0; i < M * K; i++) {
        h_a[i] = dist(gen);
    }
    float_to_fp16_to_float(h_a, h_a, M * K);
    for (int i = 0; i < K * N; i++) {
        h_b[i] = dist(gen);
    }
    float_to_fp16_to_float(h_b, h_b, K * N);

    // 在CPU上执行矩阵乘法
    std::cout << "==================================================" << std::endl;
    std::cout << "Running CPU matrix multiplication..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemm(h_a, h_b, h_c_cpu, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;
    std::cout << "CPU Time: " << cpu_elapsed.count() << " seconds" << std::endl;
    std::cout << "==================================================" << std::endl;

    half *d_a_half;
    half *d_b_half;
    float *d_c;

    cudaMalloc(&d_a_half, M * K * sizeof(half));
    cudaMalloc(&d_b_half, K * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(float));

    half *h_a_half = new half[M * K];
    half *h_b_half = new half[K * N];

    float_to_fp16(h_a_half, h_a, M * K);
    float_to_fp16(h_b_half, h_b, K * N);

    cudaMemcpy(d_a_half, h_a_half, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_half, h_b_half, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // 设置CUDA内核参数
    dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 blockDim(32, 1);  // 128线程/块

    // 清空结果矩阵
    cudaMemset(d_c, 0, M * N * sizeof(float));

    // 执行GPU矩阵乘法
    std::cout << "Running GPU Tensor Core matrix multiplication..." << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tensor_core_fp16_gemm<<<gridDim, blockDim>>>(d_a_half, d_b_half, d_c, M, N, K);
    cudaEventRecord(stop);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaEventSynchronize(stop);
    float gpu_elapsed = 0;
    cudaEventElapsedTime(&gpu_elapsed, start, stop);
    std::cout << "GPU Time: " << gpu_elapsed / 1000.0 << " seconds" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 复制结果回主机
    cudaMemcpy(h_c_gpu, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 比较结果
    std::cout << "Comparing CPU and GPU results..." << std::endl;
    compare_matrices(h_c_cpu, h_c_gpu, M * N);

    // 计算FLOPS
    double gpu_flops = (2.0 * M * N * K) / (gpu_elapsed / 1000.0);
    std::cout << "\nGPU Performance: " << gpu_flops / 1e9 << " GFLOPS" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_gpu;
    delete[] h_c_cpu;
    delete[] h_a_half;
    delete[] h_b_half;

    cudaFree(d_a_half);
    cudaFree(d_b_half);
    cudaFree(d_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


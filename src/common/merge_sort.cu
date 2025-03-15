#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <random>

void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// https://zhuanlan.zhihu.com/p/496102417


int main() {
    return 0;
}


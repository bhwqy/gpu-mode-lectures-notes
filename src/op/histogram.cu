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

__global__ void histogram_i32x4_kernel(int* a, int* y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        int4 reg_a = *reinterpret_cast<int4*>(&(a[idx]));
        atomicAdd(&(y[reg_a.x]), 1);
        atomicAdd(&(y[reg_a.y]), 1);
        atomicAdd(&(y[reg_a.z]), 1);
        atomicAdd(&(y[reg_a.w]), 1);
    }
}

__device__ void bitonic_sort(int* x) {
    constexpr static int thread_num = 1024;
    int tid = threadIdx.x;
    for (int mono_len = 2; mono_len <= thread_num; mono_len <<= 1) {
        int thread_per_mono = mono_len >> 1;
        bool ascending = (tid / thread_per_mono) % 2;
        for (int compare_offset = thread_per_mono; compare_offset > 0; compare_offset >>= 1) {
            int data_idx = (tid / compare_offset * compare_offset * 2) + (tid % compare_offset);
            if (tid < 512 && ((x[data_idx] > x[data_idx + compare_offset]) == ascending)) {
                int tmp = x[data_idx];
                x[data_idx] = x[data_idx + compare_offset];
                x[data_idx + compare_offset] = tmp;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

__device__ void reduce_start_end(short2* start_end) {
    int tid = threadIdx.x;
    for (int offset = 1; offset < 1024; offset <<= 1) {
        short2 temp = start_end[tid];
        __syncthreads();
        if (tid < 1024 - offset) {
            start_end[tid + offset].x = max(start_end[tid + offset].x, temp.x);
        }
        if (tid > offset - 1) {
            start_end[tid - offset].y = min(start_end[tid - offset].y, temp.y);
        }
        __syncthreads();
    }
}

__global__ void histogram_kernel(int* input, int* y, int N) {
    __shared__ int shared_mem[1024];
    __shared__ short2 start_end[1024];

    shared_mem[threadIdx.x] = input[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    bitonic_sort(shared_mem);
    __syncthreads();
    if (threadIdx.x == 0 || shared_mem[threadIdx.x] < shared_mem[threadIdx.x - 1]) {
        start_end[threadIdx.x].x = threadIdx.x;
    }
    else {
        start_end[threadIdx.x].x = -1;
    }
    if (threadIdx.x == 1023 || shared_mem[threadIdx.x] > shared_mem[threadIdx.x + 1]) {
        start_end[threadIdx.x].y = threadIdx.x;
    }
    else {
        start_end[threadIdx.x].y = 1024;
    }
    __syncthreads();
    reduce_start_end(start_end);
    __syncthreads();
    if (start_end[threadIdx.x].x == threadIdx.x) {
        int count = start_end[threadIdx.x].y - start_end[threadIdx.x].x + 1;
        atomicAdd(&(y[shared_mem[threadIdx.x]]), count);
    }
}

int main() {
    int times = 1024;
    int N = 1024 * times;
    int M = 1024;
    int *x = new int[N];
    int *y = new int[M];
    for (int i = 0; i < N; i++) {
        x[i] = i % 1024;
    }
    std::shuffle(x, x + N, std::random_device());

    int *d_x, *d_y;
    cudaMalloc((void**)&d_x, N * sizeof(int));
    cudaMalloc((void**)&d_y, M * sizeof(int));
    cudaMemcpy(d_x, x, N * sizeof(int), cudaMemcpyHostToDevice);

    // cudaMemset(d_y, 0, M * sizeof(int));
    // histogram_i32x4_kernel<<<1024, 256>>>(d_x, d_y, N);
    // cudaMemcpy(y, d_y, M * sizeof(int), cudaMemcpyDeviceToHost);
    // bool succ1 = true;
    // for (int i = 0; i < M; i++) {
    //     if (y[i] != 1024) {
    //         printf("Error 1: %d %d\n", i, y[i]);
    //         succ1 = false;
    //         break;
    //     }
    // }
    // if (succ1) {
    //     printf("Success 1\n");
    // }

    cudaMemset(d_y, 0, M * sizeof(int));
    histogram_kernel<<<times, 1024>>>(d_x, d_y, N);
    cudaMemcpy(y, d_y, M * sizeof(int), cudaMemcpyDeviceToHost);
    bool succ2 = true;
    for (int i = 0; i < M; i++) {
        if (y[i] != times) {
            printf("Error 2: %d %d\n", i, y[i]);
            succ2 = false;
            // break;
        }
    }
    if (succ2) {
        printf("Success 2\n");
    }

    delete[] x;
    delete[] y;

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}


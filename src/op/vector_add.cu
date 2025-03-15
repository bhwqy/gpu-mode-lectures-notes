#include <cuda_runtime.h>
#include <cstdio>

__global__ void vector_add_kernel(float *x, float *y, float *z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        z[idx] = x[idx] + y[idx];
    }
}

__global__ void vector_add_pack_kernel(float* x, float* y, float* z, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 x4 = ((float4*)x)[idx / 4];
        float4 y4 = ((float4*)y)[idx / 4];
        float4 z4;
        z4.x = x4.x + y4.x;
        z4.y = x4.y + y4.y;
        z4.z = x4.z + y4.z;
        z4.w = x4.w + y4.w;
        ((float4*)z)[idx / 4] = z4;
    }
}

int main() {
    int N = 1 << 20;
    float *x = new float[N];
    float *y = new float[N];
    float *z = new float[N];
    for (int i = 0; i < N; i++) {
        x[i] = i;
        y[i] = i;
    }
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_z, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, d_z, N);
    cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        if (z[i] != i * 2) {
            printf("Error: %f != %f\n", z[i], i * 2.0f);
            delete[] x;
            delete[] y;
            delete[] z;

            cudaFree(d_x);
            cudaFree(d_y);
            cudaFree(d_z);
            return 1;
        }
    }

    cudaMemset(d_z, 0, N * sizeof(float));
    blocks_per_grid = (N / 4 + threads_per_block - 1) / threads_per_block;
    vector_add_pack_kernel<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, d_z, N);
    cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        if (z[i] != i * 2) {
            printf("Error: %f != %f\n", z[i], i * 2.0f);
            delete[] x;
            delete[] y;
            delete[] z;

            cudaFree(d_x);
            cudaFree(d_y);
            cudaFree(d_z);
            return 2;
        }
    }

    delete[] x;
    delete[] y;
    delete[] z;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}


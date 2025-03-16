#include <cuda_runtime.h>

__global__ void mat_transpose_f32x4_shared_bcf_row2col2d_kernel(
    float *x, float *y, const int row, const int col){
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    // avoid bank conflict
    __shared__ float tile[16 * 4][16 + 1];
    if(global_y * 4 < row && global_x < col) {
        // load value from x to shared memory
        float4 x_val;
        x_val.x = x[(global_y * 4) * col + global_x];
        x_val.y = x[(global_y * 4 + 1) * col + global_x];
        x_val.z = x[(global_y * 4 + 2) * col + global_x];
        x_val.w = x[(global_y * 4 + 3) * col + global_x];
        tile[local_y * 4    ][local_x] = x_val.x;
        tile[local_y * 4 + 1][local_x] = x_val.y;
        tile[local_y * 4 + 2][local_x] = x_val.z;
        tile[local_y * 4 + 3][local_x] = x_val.w;
        __syncthreads();
        float4 smem_val;
        // load value from shared memory to y.
        // add STRIDE to satisfied different block size.
        // map index n*n to (n/4)*(n*4)
        constexpr int STRIDE = 4;
        smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
        smem_val.y = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
        smem_val.z = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
        smem_val.w = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
        const int bid_x = blockIdx.x * blockDim.x;
        const int bid_y = blockIdx.y * blockDim.y;
    
        const int out_y = bid_x + (local_y % STRIDE) * 4;
        const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
        y[out_y * row + out_x] = smem_val.x;
        y[(out_y + 1) * row + out_x] = smem_val.y;
        y[(out_y + 2) * row + out_x] = smem_val.z;
        y[(out_y + 3) * row + out_x] = smem_val.w;
    }
}



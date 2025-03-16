#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <random>

// -------------------------------------- FP32 -------------------------------------- 
// DS required for Online Softmax
struct __align__(8) MD { float m; float d; }; 
// Warp Reduce for Online Softmax
template <const int kWarpSize = 32>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        bool value_bigger = (value.m > other.m);
        MD bigger_m = value_bigger ? value : other;
        MD smaller_m = value_bigger ? other : value;
        
        value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
        value.m = bigger_m.m;
    }
    return value;
}

// Warp Reduce Sum
template <const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Warp Reduce Max
template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// grid 1D block 1D, grid(N/256), block(256)
template <int NUM_THREADS=256, int WARP_SIZE=32>
__device__ float block_reduce_sum_f32(float val) {
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
    
    float value = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    value = warp_reduce_sum_f32<NUM_WARPS>(value);  
    // WRAN: need to broadcast value to all threads within warp
    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;
}

template <int NUM_THREADS=256, int WARP_SIZE=32>
__device__ float block_reduce_max_f32(float val) {
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
    
    float value = warp_reduce_max_f32<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
    value = warp_reduce_max_f32<NUM_WARPS>(value);
    // WRAN: need to broadcast value to all threads within warp
    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;
}

// softmax implement with EXP_MAX
template<const int NUM_THREADS = 256, float EXP_MAX = 10.0f>
__global__ void softmax_f32_kernel(float* x, float* y, float* total, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid; 

    float exp_val = (idx < N) ? expf(x[idx] - EXP_MAX) : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    // get the total sum of all blocks.
    if (tid == 0) atomicAdd(total, exp_sum);
    __threadfence(); // grid level memory fence
    // e^x_i/sum(e^x_0,...,e^x_n-1) 
    // printf("N: %d, idx: %d, bid: %d, tid: %d, exp_val: %f, exp_sum: %f, total: %f\n", 
    //         N,     idx, blockIdx.x,  tid,     exp_val,     exp_sum,     *total);
    if (idx < N) y[idx] = exp_val / (*total); 
}

// online softmax
template <int NUM_THREADS = 256 / 4, int WARP_SIZE = 32>
__global__ void online_safe_softmax_f32x4_kernel(float *x, float *y, int N) {
    // reference: https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
    int local_tid = threadIdx.x;
    int global_tid = (blockIdx.x * NUM_THREADS + local_tid) * 4;

    const int WAPR_NUM = NUM_THREADS / WARP_SIZE;
    int warp_id = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;
    // compare local max value
    float4 val = *reinterpret_cast<float4*>(&(x)[global_tid]);
    float local_m = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
    float local_d = __expf(val.x - local_m) + __expf(val.y - local_m) + __expf(val.z - local_m) + __expf(val.w - local_m);

    MD local_md = {local_m, local_d};
    MD res = warp_reduce_md_op<WARP_SIZE>(local_md);
    __shared__ MD shared[WAPR_NUM];

    if (lane_id == 0) shared[warp_id] = res;
    __syncthreads();
    // do block reduce
    if (local_tid < WARP_SIZE) {
        MD block_res = shared[local_tid];
        block_res = warp_reduce_md_op<WAPR_NUM>(block_res);
        if (local_tid == 0) shared[0] = block_res;
    }
    __syncthreads();

    // TODO final res is result of only one block
    // write back
    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d);
    if (global_tid < N) {
        float4 reg_y;
        reg_y.x = __expf(val.x - final_res.m) * d_total_inverse;
        reg_y.y = __expf(val.y - final_res.m) * d_total_inverse;
        reg_y.z = __expf(val.z - final_res.m) * d_total_inverse;
        reg_y.w = __expf(val.w - final_res.m) * d_total_inverse;
        *reinterpret_cast<float4*>(&(y)[global_tid]) = reg_y;
    }
}

int main(int argc, char** argv) {
    const int N = 2048;
    float* h_A, *h_B, *h_C, *h_total;
    float* d_A, *d_B, *d_C, *d_total;

    h_A = (float*)malloc(sizeof(float) * N);
    h_B = (float*)malloc(sizeof(float) * N);
    h_C = (float*)malloc(sizeof(float) * N);
    h_total = (float*)malloc(sizeof(float));

    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_C, sizeof(float) * N);
    cudaMalloc(&d_total, sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0, 10.0);

    for (int i = 0; i < N; i++) {
        h_A[i] = dis(gen);
    }

    cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
    for (int i = 0; i < 10; ++i) {
        cudaMemset(d_total, 0, sizeof(float));
        softmax_f32_kernel<<<(N + 255) / 256, 256>>>(d_A, d_B, d_total, N);
        cudaMemcpy(h_B, d_B, sizeof(float) * N, cudaMemcpyDeviceToHost);
        online_safe_softmax_f32x4_kernel<<<(N / 4 + 255) / 256, 256>>>(d_A, d_C, N);
        cudaMemcpy(h_C, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);
        bool success = true;
        int count = 0;
        for (int i = 0; i < N; i++) {
            float abs_error = abs(h_B[i] - h_C[i]);
            float abs_sum = abs(h_B[i] + h_C[i]);
            float rel_error = abs_error / abs_sum;
            if (abs_sum > 1e-4 && rel_error > 1e-4) {
                printf("%f %f %f %f\n", h_B[i], h_C[i], abs_error, rel_error);
                success = false;
                count++;
            }
        }
        if (success) {
            printf("Success!\n");
        }
        else {
            printf("Failed! %d\n", count);
            break;
        }
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_total);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_total);

    return 0;
}


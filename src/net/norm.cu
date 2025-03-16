#include <cuda_runtime.h>

template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <int NUM_THREADS=256, int WARP_SIZE=32>
__device__ float block_reduce_sum_f32(float val) {
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
    
    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum_f32<NUM_WARPS>(val);
    return val;
}

template<const int NUM_THREADS=256/4>
__global__ void layer_norm_f32x4_kernel(float* x, float* y, float g, float b, int N, int K) {
    int tid = threadIdx.x; // 0..K-1
    int bid = blockIdx.x; // 0..N-1
    int idx = (bid * blockDim.x + threadIdx.x) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_mean; // shared within block
    __shared__ float s_variance; // shared within block
    float4 reg_x = *reinterpret_cast<float4*>(&x[idx]);
    float value = (idx < N * K) ? (reg_x.x + reg_x.y 
                                + reg_x.z + reg_x.w) : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / (float) K;
    // wait for s_mean in shared memory to be ready for all threads
    __syncthreads();
    float4 reg_x_hat;
    reg_x_hat.x = reg_x.x - s_mean;
    reg_x_hat.y = reg_x.y - s_mean;
    reg_x_hat.z = reg_x.z - s_mean;
    reg_x_hat.w = reg_x.w - s_mean;
    float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y 
                    + reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / ((float) K + epsilon));
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    float4 reg_y;
    reg_y.x = reg_x_hat.x * s_variance * g + b;
    reg_y.y = reg_x_hat.y * s_variance * g + b;
    reg_y.z = reg_x_hat.z * s_variance * g + b;
    reg_y.w = reg_x_hat.w * s_variance * g + b;
    if (idx < N * K) *reinterpret_cast<float4*>(&y[idx]) = reg_y;
}

template<const int NUM_THREADS=256/4>
__global__ void rms_norm_f32x4_kernel(float* x, float* y, float g, int N, int K) {
    int tid = threadIdx.x; // 0..K-1
    int bid = blockIdx.x; // 0..N-1
    int idx = (bid * blockDim.x + threadIdx.x) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_variance; // shared within block
    float4 reg_x = *reinterpret_cast<float4*>(&x[idx]);
    float variance = (idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y 
                                    + reg_x.z * reg_x.z + reg_x.w * reg_x.w) : 0.0f;
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads(); 
    float4 reg_y;
    reg_y.x = reg_x.x * s_variance * g;
    reg_y.y = reg_x.y * s_variance * g;
    reg_y.z = reg_x.z * s_variance * g;
    reg_y.w = reg_x.w * s_variance * g;
    if (idx < N * K) *reinterpret_cast<float4*>(y[idx]) = reg_y;
}


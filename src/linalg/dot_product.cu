#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<const int kWarpSize = 32>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
        // val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    }
    return val_f32;
}


template <const int NUM_THREADS = 256/8, int WARP_SIZE = 32>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8; // 8 half elements per thread
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    // temporary register(memory), .local space in ptx, addressable
    half pack_a[8]; // 8x16 bits=128 bits.
    // reinterpret as float4 and load 128 bits in 1 memory issue.
    *reinterpret_cast<float4*>(&pack_a[0]) = *reinterpret_cast<float4*>(&a[idx]); // load 128 bits

    float sum_f32 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum_f32 += (((idx + i ) < N) ? __half2float(pack_a[i]) : 0.0f);
    }

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    sum_f32 = warp_reduce_sum_f32<WARP_SIZE>(sum_f32);
    // warp leaders store the data to shared memory.
    // use float to keep sum from each block and reduce 
    // with fp32 inter warps.
    if (lane == 0) reduce_smem[warp] = sum_f32;
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256/8, int WARP_SIZE = 32>
__global__ void dot_prod_f16x8_pack_f32_kernel(half* a, half* b, float* y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8; // 8 half elements per thread
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    // temporary register(memory), .local space in ptx, addressable
    half pack_a[8], pack_b[8]; // 8x16 bits=128 bits.
    *reinterpret_cast<float4*>(&pack_a[0]) = *reinterpret_cast<float4*>(&a[idx]);
    *reinterpret_cast<float4*>(&pack_b[0]) = *reinterpret_cast<float4*>(&b[idx]);
    const half z = __float2half(0.0f);
    
    half prod_f16 = z;
    #pragma unroll 
    for (int i = 0; i < 8; i += 2) {
        half2 v = __hmul2(*reinterpret_cast<half2*>(&pack_a[i]),
                          *reinterpret_cast<half2*>(&pack_b[i]));
        prod_f16 += (((idx + i ) < N) ? (v.x + v.y) : z);
    }

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
    // warp leaders store the data to shared memory.
    if (lane == 0) reduce_smem[warp] = prod;
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0) atomicAdd(y, prod);
}


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


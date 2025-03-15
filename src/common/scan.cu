#include <cuda_runtime.h>

template <typename T, size_t COARSE_FACTOR = 8, size_t BLOCK_DIM = 1024>
__global__ void scan_in_block(const T* input, T* output, T* partial_sum, size_t N) {
    size_t segment = blockIdx.x * blockDim.x * COARSE_FACTOR;
    if (segment >= N) {
        return;
    }

    // load elements from global memory to shared memory
    __shared__ T buffer_s[BLOCK_DIM * COARSE_FACTOR];
    for (size_t c = 0; c < COARSE_FACTOR; ++c) {
        if (segment + c * BLOCK_DIM + threadIdx.x < N) {
            buffer_s[c * BLOCK_DIM + threadIdx.x] = input[segment + c * BLOCK_DIM + threadIdx.x];
        }
        else {
            buffer_s[c * BLOCK_DIM + threadIdx.x] = 0;
        }
    }
    __syncthreads();

    // thread scan
    size_t thread_segment = threadIdx.x * COARSE_FACTOR;
    for (size_t c = 1; c < COARSE_FACTOR; ++c) {
        buffer_s[thread_segment + c] += buffer_s[thread_segment + c - 1];
    }
    __syncthreads();

    // allocate and init double buffer for partial sum
    __shared__ T buffer1_s[BLOCK_DIM];
    __shared__ T buffer2_s[BLOCK_DIM];
    T* in_buffer_s = buffer1_s;
    T* out_buffer_s = buffer2_s;
    in_buffer_s[threadIdx.x] = buffer_s[thread_segment + COARSE_FACTOR - 1];
    __syncthreads();

    // parallel scan of partail sums
    for (size_t stride = 1; stride <= BLOCK_DIM / 2; stride <<= 1) {
        if (threadIdx.x >= stride) {
            out_buffer_s[threadIdx.x] = in_buffer_s[threadIdx.x] + in_buffer_s[threadIdx.x - stride];
        }
        else {
            out_buffer_s[threadIdx.x] = in_buffer_s[threadIdx.x];
        }
        __syncthreads();
        T* tmp = in_buffer_s;
        in_buffer_s = out_buffer_s;
        out_buffer_s = tmp;
    }

    // add previous thread's partial sum
    if (threadIdx.x > 0) {
        T prev_partial_sum = in_buffer_s[threadIdx.x - 1];
        for (size_t c = 0; c < COARSE_FACTOR; ++c) {
            buffer_s[thread_segment + c] += prev_partial_sum;
        }
    }
    __syncthreads();

    // save block's partial sum
    if (threadIdx.x == BLOCK_DIM - 1 && partial_sum != nullptr) {
        partial_sum[blockIdx.x] = in_buffer_s[threadIdx.x];
    }

    // write output
    for (size_t c = 0; c < COARSE_FACTOR; ++c) {
        if (segment + c * BLOCK_DIM + threadIdx.x < N) {
            output[segment + c * BLOCK_DIM + threadIdx.x] = buffer_s[c * BLOCK_DIM + threadIdx.x];
        }
    }
}

// https://zhuanlan.zhihu.com/p/416959273

int main() {
    return 0;
}


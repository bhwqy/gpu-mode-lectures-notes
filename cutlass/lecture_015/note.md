# CUTLASS
## Libraries that use CUTLASS
+ FlashAttention
+ Deepseek
## CUTLASS vs other CUDA libraries
+ call from host
  - CUBLAS CUDNN
  - Have a finite amount of customizability
  - Make a trip from and to device (some ability to do kernel fusion)
+ call from device
  - CUTLASS CCCL (CUB THRUST)
  - CUTLASS has low-level control, tensor core op are exposed directly
  - Code up new model, test whether and how they can be made performant

## Notations
+ We will dealing with nested tuples, an element of such a tuple (maybe tuple itself) is a mode (CUTE use)
+ Tensors in CUTE has a engine (something like pointers) and a layout consists of shape and stride.
+ indexing is (i, j, k) \dot (1, M, MN) = i + j * M + k * MN, a layout is (M, N, K) / (1, M, MN) shape / stride, shape is which inputs are allowed and stride is hot to get an coordinate from an index
+ The shape and stride must always have the same nesting structure. (They must be congruent in CUTE term).
+ Function composition for layout and sub-layout. with-shape local-tile local-partition 
+ Non-contiguous
+ Slicing and tiling

## CODE
### CUTE
#### ARCH ATOM
Example of mma_sm80.hpp

SM80_16x8x32_S32S8S8S32_TN
T represents transpose and N represents no transpose

asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));

The first part is string concatenation like a template. The second part is a list of output registers. Output registers alway begins with =.
The third part is a list of input registers.

using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

gives How many registers are used in the code?

Example of mma_traits_sm80.hpp

template <>
struct MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = SM80_16x8_Row;
  using BLayout = SM80_8x8_Row;
  using CLayout = SM80_16x8_Row;
};

Gives thread id, layout of A, B, C matrices

tiled_copy.cu

Things like inverse of a matrix is not implemented in CUTLASS and may be found in cuSolver and cuBLAS.


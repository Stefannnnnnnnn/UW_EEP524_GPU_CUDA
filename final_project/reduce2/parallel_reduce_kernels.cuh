
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// convenient defines
#define uchar unsigned char
#define uint unsigned int

#define tz threadIdx.z
#define ty threadIdx.y
#define tx threadIdx.x
#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

//=======================================================================================

__global__ void reduce_sum_poor_gmem_multiblock_kernel(float* input, float* output)
{
    uint seg_offset = 2 * blockDim.x * bx;
    uint i = seg_offset + 2 * tx;  // initial assignment of thread-owned indices

    for (uint stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if (tx % stride == 0)  // control+mem divergence - participating threads increasingly far apart
        {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (tx == 0)
    {
        atomicAdd(output, input[seg_offset]);
    }
}

//=======================================================================================

__global__ void reduce_sum_improved_gmem_multiblock_kernel(float* input, float* output)
{
    uint seg_offset = 2 * blockDim.x * bx;
    uint i = seg_offset + tx;  // initial assignment of thread-owned indices

    for (uint stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (tx < stride)  // improved control+mem divergence - participating threads use left-side compaction
        {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (tx == 0)
    {
        atomicAdd(output, input[seg_offset]);
    }
}

//=======================================================================================

// ADD Shared Memory usage
#define SEG_BLOCK_DIM 1024
__global__ void reduce_sum_segmented_multiblock_kernel(float* input, float* output)
{
    __shared__ float input_s[SEG_BLOCK_DIM];

    uint seg_offset = 2 * blockDim.x * blockIdx.x;
    uint i = seg_offset + tx;  // initial assignment of per-segment thread-owned indices

    input_s[tx] = input[i] + input[i + SEG_BLOCK_DIM]; // first reduction pass use GMEM and move into SMEM

    for (uint stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads(); // wait for all data from first reduction pass using GMEM to get into SMEM
        if (tx < stride)  // participating threads grouped on left side
        {
            input_s[tx] += input_s[tx + stride];
        }
    }

    if (tx == 0)
    {
        atomicAdd(output, input_s[0]);
    }
}

//=======================================================================================

#define CFACT 2   // number of original thread blocks which coarsened block does work of
#define COARSE_BLOCK_DIM 512
__global__ void reduce_sum_segmented_coarsened_multiblock_kernel(float* input, float* output)
{
    __shared__ float input_s[COARSE_BLOCK_DIM];

    uint seg_offset = CFACT * 2 * blockDim.x * blockIdx.x;
    uint i = seg_offset + tx;  // initial assignment of per-segment thread-owned indices

    float sum = input[i];
    for (uint tile = 1; tile < CFACT * 2; ++tile)  // coarsened reduction loop for coarsened tiles
    {
        sum += input[i + tile * COARSE_BLOCK_DIM];
    }
    input_s[tx] = sum;

    for (uint stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads(); // wait for all data from first reduction pass using GMEM to get into SMEM
        if (tx < stride)  // participating threads grouped on left side
        {
            input_s[tx] += input_s[tx + stride];
        }
    }

    if (tx == 0)
    {
        atomicAdd(output, input_s[0]);
    }
}

//=======================================================================================


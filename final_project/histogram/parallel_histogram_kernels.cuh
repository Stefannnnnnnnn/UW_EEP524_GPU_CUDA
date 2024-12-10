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
//
// KTC-NOTE: Do not change NUM_BINS and BIN_WIDTH
//
#define NUM_BINS 26  // one per alphabet letter
#define BIN_WIDTH 1

//=======================================================================================

__global__ void histo_basic_kernel(char* data, uint length, uint* histo)
{
    uint i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < length)
    {
        int alpha_pos = data[i] - 'a';
        if (alpha_pos >= 0 && alpha_pos < 26)
        {
            atomicAdd(&(histo[alpha_pos / BIN_WIDTH]), (uint)1);
        }
    }
}

//=======================================================================================
// KTC-NOTE: This kernel is having memory corruption issues with large input sizes. 
// If you can identify the issue and fix it (BONUS PTS), feel free to include, otherwise omit.
// Problem was isolated to the atomicAdd calls - they are currently commented out
// It is possible that the # of atomic units in GMEM and/or L2 is being exceeded.
//
// // remember this will probably use L2 cache atomics
__global__ void histo_priv_global_kernel(char* data, uint length, uint numBins, uint* histo)
{
    uint i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < length)
    {
        int alpha_pos = data[i] - 'a';
        if (alpha_pos >= 0 && alpha_pos < 26)
        {
            //atomicAdd(&(histo[0]), (uint)1); //DEBUG
            //atomicAdd(&(histo[blockIdx.x * numBins + alpha_pos / BIN_WIDTH]), (uint)1);
        }
    }

    // merge private copy into master copy (block 0)
    if (blockIdx.x > 0)
    {
        __syncthreads();
        for (uint bin = tx; bin < numBins; bin += blockDim.x)
        {
            uint binVal = histo[blockIdx.x * numBins + bin]; // read value from private histo region
            if (binVal > 0)
            {
                //atomicAdd(&(histo[0]), binVal); // update value in master histogram copy //DEBUG
                //atomicAdd(&(histo[bin]), binVal); // update value in master histogram copy
            }
        }
    }
}

//=======================================================================================

__global__ void histo_priv_smem_kernel(char* data, uint length, uint* histo)
{
    __shared__ uint histo_s[NUM_BINS];
    //collaborative init (zero) of smem
    for (uint bin = tx; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();

    // accumulate segment into SM SMEM private histogram
    uint i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < length)
    {
        int alpha_pos = data[i] - 'a';
        if (alpha_pos >= 0 && alpha_pos < 26)
        {
            atomicAdd(&(histo_s[alpha_pos / BIN_WIDTH]), (uint)1);
        }
    }
    __syncthreads();

    // ALL blocks merge private SMEM copy into master  GMEM copy
    for (uint bin = tx; bin < NUM_BINS; bin += blockDim.x)
    {
        uint binval = histo_s[bin];
        if (binval > 0)
        {
            atomicAdd(&(histo[bin]), binval);
        }
    }
}

//=======================================================================================

__global__ void histo_coarsen_interleaved_kernel(char* data, uint length, uint* histo)
{
    __shared__ uint histo_s[NUM_BINS];
    //collaborative init (zero) of smem privatized bins
    for (uint bin = tx; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();

    // coarsened accumulate into SM SMEM histogram
    // threads are interleaved, stride by entire grid each iteration
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (uint i = tid; i < length; i += blockDim.x * gridDim.x)
    {
        int alpha_pos = data[i] - 'a';
        if (alpha_pos >= 0 && alpha_pos < 26)
        {
            atomicAdd(&(histo_s[alpha_pos / BIN_WIDTH]), (uint)1);
        }
    }
    __syncthreads();

    // ALL blocks commit private SMEM copy into master  GMEM copy
    for (uint bin = tx; bin < NUM_BINS; bin += blockDim.x)
    {
        uint binval = histo_s[bin];
        if (binval > 0)
        {
            atomicAdd(&(histo[bin]), binval);
        }
    }
}

//=======================================================================================

__global__ void histo_coarsen_contiguous_kernel(char* data, uint length, uint* histo, uint CFACTOR)
{
    // CFACTOR defines
    __shared__ uint histo_s[NUM_BINS];
    //collaborative init (zero) of smem privatized bins
    for (uint bin = tx; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();

    // coarsened accumulate into SM SMEM histogram
    // threads use contiguous partitioning, stride by one for width of CFACTOR
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (uint i = tid * CFACTOR; i < min(length, (tid + 1) * CFACTOR); ++i)
    {
        int alpha_pos = data[i] - 'a';
        if (alpha_pos >= 0 && alpha_pos < 26)
        {
            atomicAdd(&(histo_s[alpha_pos / BIN_WIDTH]), (uint)1);
        }
    }
    __syncthreads();

    // ALL blocks commit private SMEM copy into master  GMEM copy
    for (uint bin = tx; bin < NUM_BINS; bin += blockDim.x)
    {
        uint binval = histo_s[bin];
        if (binval > 0)
        {
            atomicAdd(&(histo[bin]), binval);
        }
    }
}

//=======================================================================================
// Host App CPU reference sequential
void histo_seq_host_cpu(char* data, uint length, uint* histo)
{
    for (uint i = 0; i < length; ++i)
    {
        int alpha_pos = data[i] - 'a';
        // only include lower-case alphabet chars, skip all others
        // use 2-wide bins (total of 13 bins)
        if (alpha_pos >= 0 && alpha_pos < 26)
            histo[alpha_pos / BIN_WIDTH]++;
    }
}

//=======================================================================================


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

//============================================================================
// KTC-NOTE: DO NOT CHANGE THE 7-PT STENCIL DEFINITIONS
#define STENCIL_RADIUS_3D 1  // all 3D stencil kernels assume same stencil radius!

// 3D 7-pt stencil coefficients, same layout/ordering as device coeffs below
float h_7pt_stencil_coeffs[] = { 2.0f,1.0f,2.0f,1.5f,2.5f,1.75f,2.33f };

// NOTE: these could also be placed in __contant__ or __shared__ or even possibly in thread registers
__device__ float c0 = 2.0f; // centerpoint stencil cell (x,y,z)
__device__ float c1 = 1.0f;  // x-1
__device__ float c2 = 2.0f;  // x+1
__device__ float c3 = 1.5f;  // y-1
__device__ float c4 = 2.5f;  // y+1
__device__ float c5 = 1.75f;  // z-1
__device__ float c6 = 2.33f;  // Z+1

//__device__ float c0 = 1.0f; // centerpoint stencil cell (x,y,z)
//__device__ float c1 = 1.0f;  // x-1
//__device__ float c2 = 1.0f;  // x+1
//__device__ float c3 = 1.0f;  // y-1
//__device__ float c4 = 1.0f;  // y+1
//__device__ float c5 = 1.0f;  // z-1
//__device__ float c6 = 1.0f;  // Z+1
//
//__device__ float cX = 7.7f;  // INVALID DEBUG TEST COEFF

//============================================================================

// Assumes uniform NxNxN cubic 3D data INPut grid
__global__ void stencil_3d_basic_kernel(float* dIN, float* dOUT, uint N)
{
    uint i = blockIdx.z * blockDim.z + threadIdx.z; // slice dim
    uint j = blockIdx.y * blockDim.y + threadIdx.y; // row dim
    uint k = blockIdx.x * blockDim.x + threadIdx.x; // col dim = fastest moving

    //printf("<3D basic stencil>threadID: (%d,%d,%d)\n", k, j, i);

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1)  // skip outermost 3D grid edge boundary-condition cells
    {
        // compute output element by applying 3D stencil coefficients to input elements
        dOUT[i * N * N + j * N + k] = c0 * dIN[i * N * N + j * N + k]
            + c1 * dIN[i * N * N + j * N + (k - 1)]
            + c2 * dIN[i * N * N + j * N + (k + 1)]
            + c3 * dIN[i * N * N + (j - 1) * N + k]
            + c4 * dIN[i * N * N + (j + 1) * N + k]
            + c5 * dIN[(i - 1) * N * N + j * N + k]
            + c6 * dIN[(i + 1) * N * N + j * N + k];
        // printf("<3D basic stencil>threadID: (%d,%d,%d) valid output cell: OUT = %d\n", k, j, i, OUT[i * N * N + j * N + k]);
    }
}


//============================================================================

// all 3D stencil kernels assume same stencil radius! (see top)
#define IN_TILE_DIM 8
#define OUT_TILE_DIM (IN_TILE_DIM - 2*STENCIL_RADIUS_3D)

// due to 3D thread block sizes of IN_TILE_DIM^3 the HW limit is 8x8x8=512 CTAs
// thus for a cube grid of NxNxN will need  ceil( (N-2) / (OUT_TILE_DIM) ) grid in each DIM
// Assumes uniform NxNxN cubic 3D data INPut grid
__global__ void stencil_3d_tiled_smem_kernel(float* dIN, float* dOUT, uint N)
{
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    //uint i = blockIdx.z * OUT_TILE_DIM + tz - STENCIL_RADIUS_3D;
    //uint j = blockIdx.y * OUT_TILE_DIM + ty - STENCIL_RADIUS_3D;
    //uint k = blockIdx.x * OUT_TILE_DIM + tx - STENCIL_RADIUS_3D;

    // Corrected indexing from PMPP due to output tile size striding offset by +1
    // NOTE this may not work for stencils w radius != 1
    // these i,j,k values should only include valid interior output-grid elements to be computed & output
    uint i = blockIdx.z * OUT_TILE_DIM + tz;
    uint j = blockIdx.y * OUT_TILE_DIM + ty;
    uint k = blockIdx.x * OUT_TILE_DIM + tx;

    //printf("<3D tiled SMEM stencil> (x,y,z) blockID(%d,%d,%d) threadID: (%d,%d,%d)\n", i, j, k, tx, ty, tz);
    // All threads in block collaboratively move all valid 3D grid points GMEM -> SMEM
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N)
    {
        //printf("<3D tiled SMEM stencil> LOADING TO SMEM:  (x,y,z) blockID(%d,%d,%d) threadID: (%d,%d,%d)\n", i, j, k, tx, ty, tz);
        in_s[tz][ty][tx] = dIN[i * N * N + j * N + k];
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1)  // skip outermost 3D grid edge boundary-condition cells
    {
        // disable threads which only performed input cell memory movement - only allow threads within output tile
        if (tz >= STENCIL_RADIUS_3D && tz < IN_TILE_DIM - STENCIL_RADIUS_3D && ty >= STENCIL_RADIUS_3D && ty < IN_TILE_DIM - STENCIL_RADIUS_3D && tx >= STENCIL_RADIUS_3D && tx < IN_TILE_DIM - STENCIL_RADIUS_3D)
        {
            //printf("<3D tiled SMEM stencil> CALC OUT[] : (x,y,z) blockID(%d,%d,%d) threadID: (%d,%d,%d)\n", i, j,k, tx, ty, tz);
            dOUT[i * N * N + j * N + k] = c0 * in_s[tz][ty][tx]
                + c1 * in_s[tz][ty][tx - 1]
                + c2 * in_s[tz][ty][tx + 1]
                + c3 * in_s[tz][ty - 1][tx]
                + c4 * in_s[tz][ty + 1][tx]
                + c5 * in_s[tz - 1][ty][tx]
                + c6 * in_s[tz + 1][ty][tx];
        }
    }

}

//============================================================================
//
// NON-SQUARE PROCESSING THREADBLOCK = TILE DIMS (for better GMEM COALESCING)
// all 3D stencil kernels assume same stencil radius! (see top)
#define IN_TILE_DIM_X 32
#define IN_TILE_DIM_YZ 4
#define OUT_TILE_DIM_X (IN_TILE_DIM_X - 2*STENCIL_RADIUS_3D)
#define OUT_TILE_DIM_YZ (IN_TILE_DIM_YZ - 2*STENCIL_RADIUS_3D)

// due to 3D thread block sizes of IN_TILE_DIM^3 the HW limit is 8x8x8=512 CTAs
// thus for a cube grid of NxNxN will need  ceil( (N-2) / (OUT_TILE_DIM) ) grid in each DIM
// Assumes uniform NxNxN cubic 3D data INPut grid
__global__ void stencil_3d_nonsqr_tiled_smem_kernel(float* dIN, float* dOUT, uint N)
{
    __shared__ float in_s[IN_TILE_DIM_YZ][IN_TILE_DIM_YZ][IN_TILE_DIM_X];

    //uint i = blockIdx.z * OUT_TILE_DIM + tz - STENCIL_RADIUS_3D;
    //uint j = blockIdx.y * OUT_TILE_DIM + ty - STENCIL_RADIUS_3D;
    //uint k = blockIdx.x * OUT_TILE_DIM + tx - STENCIL_RADIUS_3D;

    // Corrected indexing from PMPP due to output tile size striding offset by +1
    // NOTE this may not work for stencils w radius != 1
    // these i,j,k values should only include valid interior output-grid elements to be computed & output
    uint i = blockIdx.z * OUT_TILE_DIM_YZ + tz;
    uint j = blockIdx.y * OUT_TILE_DIM_YZ + ty;
    uint k = blockIdx.x * OUT_TILE_DIM_X + tx;

    //printf("<3D tiled SMEM stencil> (x,y,z) blockID(%d,%d,%d) threadID: (%d,%d,%d)\n", i, j, k, tx, ty, tz);
    // All threads in block collaboratively move all valid 3D grid points GMEM -> SMEM
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N)
    {
        //printf("<3D tiled SMEM stencil> LOADING TO SMEM:  (x,y,z) blockID(%d,%d,%d) threadID: (%d,%d,%d)\n", i, j, k, tx, ty, tz);
        in_s[tz][ty][tx] = dIN[i * N * N + j * N + k];
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1)  // skip outermost 3D grid edge boundary-condition cells
    {
        // disable threads which only performed input cell memory movement - only allow threads within output tile
        if (tz >= STENCIL_RADIUS_3D && tz < IN_TILE_DIM_YZ - STENCIL_RADIUS_3D && ty >= STENCIL_RADIUS_3D && ty < IN_TILE_DIM_YZ - STENCIL_RADIUS_3D && tx >= STENCIL_RADIUS_3D && tx < IN_TILE_DIM_X - STENCIL_RADIUS_3D)
        {
            //printf("<3D tiled SMEM stencil> CALC OUT[] : (x,y,z) blockID(%d,%d,%d) threadID: (%d,%d,%d)\n", i, j,k, tx, ty, tz);
            dOUT[i * N * N + j * N + k] = c0 * in_s[tz][ty][tx]
                + c1 * in_s[tz][ty][tx - 1]
                + c2 * in_s[tz][ty][tx + 1]
                + c3 * in_s[tz][ty - 1][tx]
                + c4 * in_s[tz][ty + 1][tx]
                + c5 * in_s[tz - 1][ty][tx]
                + c6 * in_s[tz + 1][ty][tx];
        }
    }

}


//============================================================================
//
// all 3D stencil kernels assume same stencil radius! (see top)
const int IN_TILE_DIMB = 32;
const int OUT_TILE_DIMB = (IN_TILE_DIMB - 2 * STENCIL_RADIUS_3D);
//
// since this kernel uses 2D threadblocks to iterate in Z-dim by OUT_TILE_DIMB for each block
// and thus the XY threadblock sizes can be larger (max 32x32 = 1024 threads/block)
// the launch configuration will need to be
// 
// for a cube grid of NxNxN 
// will need  ceil( (N) / (OUT_TILE_DIMB) ) grid in Z DIM
//
// Assumes uniform NxNxN cubic 3D data INPut grid
__global__ void stencil_3d_thread_coarsening_kernel(float* dIN, float* dOUT, uint N)
{
    __shared__ float inPrev_s[IN_TILE_DIMB][IN_TILE_DIMB];
    __shared__ float inCurr_s[IN_TILE_DIMB][IN_TILE_DIMB];
    __shared__ float inNext_s[IN_TILE_DIMB][IN_TILE_DIMB];

    uint iStart = blockIdx.z * OUT_TILE_DIMB;
    uint j = blockIdx.y * OUT_TILE_DIMB + ty; // -STENCIL_RADIUS_3D;
    uint k = blockIdx.x * OUT_TILE_DIMB + tx; // -STENCIL_RADIUS_3D;

    // threads collaboratively iteratively load valid 3D grid points GMEM -> SMEM
    // into 2D SMEM slices for thread-coarsened algorithm
    // NOTE: boundary check conditions have been combined for inPrev & inCurr : deviates from PMPP kernel
    if (iStart >= 0 && iStart < N - 1 && j >= 0 && j < N && k >= 0 && k < N)
    {
        inPrev_s[ty][tx] = dIN[(iStart)*N * N + j * N + k];
        inCurr_s[ty][tx] = dIN[(iStart + 1) * N * N + j * N + k];
    }
    else
    {
        inPrev_s[ty][tx] = 0;
        inCurr_s[ty][tx] = 0;
    }

    for (int i = iStart + 1; i <= iStart + OUT_TILE_DIMB; ++i) // main coarsened loop
    {
        // collaboratively & interatively move Next input XY-plane elements into SMEM
        if (i >= 1 && i < N - 1 && j >= 0 && j < N && k >= 0 && k < N)
        {
            inNext_s[ty][tx] = dIN[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();

        // now calc only valid output elements, excluding boundary cells (1st and last values in each dim)
        if (i >= 1 && i < (N - 1) && j >= 1 && j < (N - 1) && k >= 1 && k < (N - 1))
        {
            if (ty >= 1 && ty < IN_TILE_DIMB - 1 && tx >= 1 && tx < IN_TILE_DIMB - 1)
            {
                dOUT[i * N * N + j * N + k] = c0 * inCurr_s[ty][tx]
                    + c1 * inCurr_s[ty][tx - 1]
                    + c2 * inCurr_s[ty][tx + 1]
                    + c3 * inCurr_s[ty - 1][tx]
                    + c4 * inCurr_s[ty + 1][tx]
                    + c5 * inPrev_s[ty][tx]
                    + c6 * inNext_s[ty][tx];
            }
        }
        __syncthreads();

        inPrev_s[ty][tx] = inCurr_s[ty][tx];
        inCurr_s[ty][tx] = inNext_s[ty][tx];
    }
}

//============================================================================



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <string>

#include "windows.h" // include for QueryPerfTimer API instead of profile.h
#include "parallel_reduce_kernels.cuh"

using namespace std;
#define POOR_GMEM 0
#define IMPROVED_GMEM 1
#define SEGMENTED 2
#define COARSENED 3

// for Win API QPC host timing
LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
LARGE_INTEGER Frequency;


//#define STB_IMAGE_IMPLEMENTATION
//#include "../util/stb_image.h"  // download from class website files
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "../util/stb_image_write.h"  // download from class website files


// #include your error-check macro header file here
static cudaError_t cudartCHK(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        printf("CUDA Runtime API Error: %s:%d\n", __FILE__, __LINE__);
        printf("error code:%d, name: %s reason: %s\n", result, cudaGetErrorName(result), cudaGetErrorString(result));
    }
    return result;
}

int read_binary_grid_file(string strFile, char** memblock)
{
    streampos size;

    ifstream file(strFile, ios::in | ios::binary | ios::ate);
    if (file.is_open())
    {
        size = file.tellg();
        *memblock = new char[size];
        file.seekg(0, ios::beg);
        file.read(*memblock, size);
        file.close();

        cout << "Entire file content of file: " << strFile << " is in memory of size: " << (int)size << " bytes." << endl;
        return (int)size;
    }
    else
    {
        cout << "Unable to open binary file: " << strFile << endl;
        return NULL;
    }
}

int write_bin_grid_file(string strFile, char* memblock, uint numBytes)
{
    ofstream file(strFile, ios::out | ios::binary | ios::trunc);
    if (file.is_open())
    {
        file.write(memblock, numBytes);
        file.close();

        cout << "Binary data has been written to file " << strFile << endl;
        return 0;
    }
    else
    {
        cout << "Unable to write binary file: " << strFile << endl;
        return -1;
    }

}

// ============================================================================================
// Host App GPU verification sequential routines

float reduce_sum_seq_host_cpu(float* data, uint length)
{
    double result = 0.0;  // each reduce may have different Identity definition for init.

    for (uint i = 0; i < length; ++i)
    {
        result += data[i];
    }
    return (float)result;
}

double reduce_sum_seq_host_cpu_double(float* data, uint length)
{
    double result = 0.0;  // each reduce may have different Identity definition for init.

    for (uint i = 0; i < length; ++i)
    {
        result += data[i];
    }
    return result;
}

// ============================================================================================
// SINGLE-BLOCK Kernels (only work for a single threadblock)
// KTC-NOTE: DO NOT USE THESE FOR KTC

__global__ void reduce_sum_basic_singleblock_kernel(float* input, float* output)
{
    uint i = 2 * tx;  // initial assignment of thread-owned indices

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
        *output = input[0];
    }
}

// this version uses left-sided compaction of the active threads for each reduction
// improves both control divergence and memory coalescing
__global__ void reduce_sum_improved_singleblock_kernel(float* input, float* output)
{
    uint i = tx;  // initial assignment of thread-owned indices

    for (uint stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (tx < stride)  // participating threads grouped on left side
        {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (tx == 0)
    {
        *output = input[0];
    }
}

#define BLOCK_DIM 1024
__global__ void reduce_sum_smem_singleblock_kernel(float* input, float* output)
{
    __shared__ float input_s[BLOCK_DIM];
    uint i = tx;  // initial assignment of thread-owned indices

    // threads do coalesced initial sum into SMEM
    input_s[i] = input[i] + input[i + BLOCK_DIM];

    for (uint stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads(); // all threads must reach sync point
        if (tx < stride)  // participating threads grouped on left side
        {
            input_s[i] += input_s[i + stride];
        }
    }

    __syncthreads();
    if (tx == 0)
    {
        *output = input_s[0];
        //atomicAdd(output, input_s[0]); // HACK: attempt to make multi-block
    }
}

// ============================================================================================



int main(int argc, char** argv)
{
    //load float numeric string to reduce
    string strInFile = "../dat/reduce_all-ones_n1048576_float32.bin";

    char* memblk = NULL;
    int charDataBytes = read_binary_grid_file(strInFile, &memblk);
    float* h_IN = (float*)memblk;

    double* h_INdbl = (double*)malloc(2 * charDataBytes);

    int numFloats = charDataBytes / sizeof(float);

    //// setup host variables, allocate host memory as needed
    float h_OUT1 = 0.0f;
    float h_OUT2 = 0.0f;
    float h_OUT3 = 0.0f;
    float h_OUT4 = 0.0f;
    float h_OUT5 = 0.0f;
    float h_OUT6 = 0.0f;

    double h_OUT1dbl = 0;
    double h_OUT2dbl = 0;

    //uint* h_OUT1 = (uint*)malloc(numHistoBins * sizeof(uint));
    //memset(h_OUT1, 0, numHistoBins * sizeof(uint));

    //=====================================================================================================
    // HOST-side reference verification sequential routine timing

    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);

    // run CPU sequential reference
    h_OUT1 = reduce_sum_seq_host_cpu(h_IN, numFloats);

    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

    // report CPU Ref timing
    cout << "CPU Reference sum-reduction sequential time: " << ElapsedMicroseconds.QuadPart << " (us)" << endl;
    //=====================================================================================================

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudartCHK(cudaSetDevice(0));

    // allocate device memory
    float* d_IN = 0;
    float* d_OUT = 0;
    double* d_INdbl = 0;
    double* d_OUTdbl = 0;
    cudartCHK(cudaMalloc((void**)&d_IN, charDataBytes));
    cudartCHK(cudaMalloc((void**)&d_OUT, sizeof(float)));

    //=====================================================================================================
    // HOST-side GPU sequential routine timing
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);

    // Copy input data from host memory to GPU buffers.
    cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice));
    cudartCHK(cudaMemset(d_OUT, 0, sizeof(float)));  // since using atomic adds
    // Launch a 1D kernel on the GPU with specified launch configuration dimensions
    dim3 dimBlock2D(SEG_BLOCK_DIM, 1, 1);
    dim3 dimGrid2D(numFloats / 2 / SEG_BLOCK_DIM, 1, 1);
    const int kernel_type = COARSENED;

    if (kernel_type == POOR_GMEM) {
        // ==== KERNEL Ver-1
        reduce_sum_poor_gmem_multiblock_kernel << <dimGrid2D, dimBlock2D >> > (d_IN, d_OUT);
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudartCHK(cudaDeviceSynchronize());
        // Check for any errors launching the kernel
        cudartCHK(cudaGetLastError());
        // retrieve result data from device back to host
        // Copy output image from GPU buffer to host memory.
        cudartCHK(cudaMemcpy(&h_OUT2, d_OUT, sizeof(float), cudaMemcpyDeviceToHost));
        cudartCHK(cudaDeviceSynchronize());
    }
    else if (kernel_type == IMPROVED_GMEM) {
        // ==== KERNEL Ver-2
        cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice));
        cudartCHK(cudaMemset(d_OUT, 0, sizeof(float)));
        cudartCHK(cudaDeviceSynchronize());
        reduce_sum_improved_gmem_multiblock_kernel << <dimGrid2D, dimBlock2D >> > (d_IN, d_OUT);
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudartCHK(cudaDeviceSynchronize());
        // Check for any errors launching the kernel
        cudartCHK(cudaGetLastError());
        // retrieve result data from device back to host
        // Copy output image from GPU buffer to host memory.
        cudartCHK(cudaMemcpy(&h_OUT3, d_OUT, sizeof(float), cudaMemcpyDeviceToHost));
        cudartCHK(cudaDeviceSynchronize());
    }
    // // ==== KERNEL Ver-3 - SMEM singleblock
    //cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice));
    //cudartCHK(cudaMemset(d_OUT, 0, sizeof(float)));
    //cudartCHK(cudaDeviceSynchronize());
    //reduce_sum_smem_singleblock_kernel << <dimGrid2D, dimBlock2D >> > (d_IN, d_OUT);
    // // cudadevicesynchronize waits for the kernel to finish, and returns
    // // any errors encountered during the launch.
    //cudartCHK(cudaDeviceSynchronize());
    //// Check for any errors launching the kernel
    //cudartCHK(cudaGetLastError());
    // // retrieve result data from device back to host
    //// copy output image from gpu buffer to host memory.
    //cudartCHK(cudaMemcpy(&h_OUT4, d_OUT, sizeof(float), cudaMemcpyDeviceToHost));
    //cudartCHK(cudaDeviceSynchronize());
    else if (kernel_type == SEGMENTED) {
        // ==== KERNEL Ver-4 = Segmented multi-block
        cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice));
        cudartCHK(cudaMemset(d_OUT, 0, sizeof(float)));
        cudartCHK(cudaDeviceSynchronize());
        reduce_sum_segmented_multiblock_kernel << <dimGrid2D, dimBlock2D >> > (d_IN, d_OUT);
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudartCHK(cudaDeviceSynchronize());
        // Check for any errors launching the kernel
        cudartCHK(cudaGetLastError());
        // retrieve result data from device back to host
       // Copy output image from GPU buffer to host memory.
        cudartCHK(cudaMemcpy(&h_OUT5, d_OUT, sizeof(float), cudaMemcpyDeviceToHost));
        cudartCHK(cudaDeviceSynchronize());
    }
    else if (kernel_type == COARSENED) {
        // ==== KERNEL Ver-5 = Coarsened segmented multi-block
        cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice));
        cudartCHK(cudaMemset(d_OUT, 0, sizeof(float)));
        cudartCHK(cudaDeviceSynchronize());
        dim3 dimCoarseBlock2D(1, 1, 1);
        dim3 dimCoarseGrid2D(1, 1, 1);
        dimCoarseBlock2D.x = COARSE_BLOCK_DIM;
        dimCoarseGrid2D.x = numFloats / 2 / CFACT / dimCoarseBlock2D.x;
        reduce_sum_segmented_coarsened_multiblock_kernel << <dimCoarseGrid2D, dimCoarseBlock2D >> > (d_IN, d_OUT);
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudartCHK(cudaDeviceSynchronize());
        // Check for any errors launching the kernel
        cudartCHK(cudaGetLastError());
        // retrieve result data from device back to host
        // Copy output image from GPU buffer to host memory.
        cudartCHK(cudaMemcpy(&h_OUT6, d_OUT, sizeof(float), cudaMemcpyDeviceToHost));
        cudartCHK(cudaDeviceSynchronize());
    }

    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

    // report CPU Ref timing
    cout << "GPU sum-reduction sequential time: " << ElapsedMicroseconds.QuadPart << " (us)" << endl;
    //=====================================================================================================

    cudartCHK(cudaDeviceReset()); // must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    // verify results
    //printf("VERIF: 1D gridsize: # blocks = %d #threads/block = %d\n", dimGrid2D.x, dimBlock2D.x);
    //printf("VERIF: 1D COARSENED gridsize: # blocks = %d #threads/block = %d\n", dimCoarseGrid2D.x, dimCoarseBlock2D.x);
    //printf("VERIFICATION (single-block SMEM!): CPU reference h_OUT1 and GPU h_OUT4 reduction results: %f, %f\n", h_OUT1, h_OUT4);
    printf("VERIFICATION: CPU reference h_OUT1 and GPU h_OUT2 reduction results: %f, %f\n", h_OUT1, h_OUT2);
    printf("VERIFICATION: CPU reference h_OUT1 and GPU h_OUT3 reduction results: %f, %f\n", h_OUT1, h_OUT3);
    printf("VERIFICATION: CPU reference h_OUT1 and GPU h_OUT5 reduction results: %f, %f\n", h_OUT1, h_OUT5);
    printf("VERIFICATION: CPU reference h_OUT1 and GPU h_OUT6 reduction results: %f, %f\n", h_OUT1, h_OUT6);

    // free host and device memory
    free(h_INdbl);
    cudaFree(d_IN);
    cudaFree(d_OUT);

    return 0;
}




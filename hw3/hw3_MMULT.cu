#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <time.h>
#include "../util/check_error.h"

#define TILE_WIDTH 32
#define NAIVE 0
#define SHARED 1

// CPU reference implementation for verification
void matMulCPU(const float* M, const float* N, float* P, int j, int k, int l) {
    float sum = 0;
    for (int i = 0; i < j; ++i) {
        for (int col = 0; col < l; ++col) {
            sum = 0;
            for (int width = 0; width < k; ++width) {
                sum += M[i * k + width] * N[width * l + col];
            }
            P[i * l + col] = sum;
        }
    }
}

// Naive GPU kernel with only global memory
__global__ void matMulKernelNaive(const float* M, const float* N, float* P, int j, int k, int l) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < j && col < l) {
        float sum = 0;
        for (int width = 0; width < k; ++width) {
            sum += M[row * k + width] * N[width * l + col];
        }
        P[row * l + col] = sum;
    }
}

// Optimized GPU kernel using shared memory tiling
__global__ void matMulKernelShared(const float* M, const float* N, float* P, int j, int k, int l) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    // Loop over tiles
    for (int m = 0; m < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load elements into shared memory with boundary check
        if (Row < j && m * TILE_WIDTH + tx < k) {
            Mds[ty][tx] = M[Row * k + m * TILE_WIDTH + tx];
        }
        else {
            Mds[ty][tx] = 0.0;
        }
        if (Col < l && m * TILE_WIDTH + ty < k) {
            Nds[ty][tx] = N[(m * TILE_WIDTH + ty) * l + Col];
        }
        else {
            Nds[ty][tx] = 0.0;
        }
        __syncthreads();

        // Multiply tile elements
        for (int e = 0; e < TILE_WIDTH; ++e) {
            Pvalue += Mds[ty][e] * Nds[e][tx];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (Row < j && Col < l) {
        P[Row * l + Col] = Pvalue;
    }
}

// Verification function
void verifyResults(const float* P_cpu, const float* P_gpu, int size) {
    double threshold = 1.0f;
    float max_error = 0.0;
    for (int i = 0; i < size; i++) {
        float error = fabs(P_cpu[i] - P_gpu[i]) / fabs(P_cpu[i]);
        if (error > max_error)
            max_error = error;
    }
    for (int i = 0; i < 300; i++, threshold *= 0.1) {
        if (max_error >= threshold) {
            printf("Maximum error fraction >= 10^(-x), x = %d\n", i);
            return;
        }
    }
    printf("no error fractions observed\n");
}

// Host function
void matMulGPU(const float* M, const float* N, float* P, int j, int k, int l) {
    checkCuda(cudaSetDevice(0));
    
    // Allocate GPU memory
    float* dev_M, * dev_N, * dev_P;
    size_t sizeM = j * k * sizeof(float);
    size_t sizeN = k * l * sizeof(float);
    size_t sizeP = j * l * sizeof(float);

    checkCuda(cudaMalloc((void**)&dev_M, sizeM));
    checkCuda(cudaMalloc((void**)&dev_N, sizeN));
    checkCuda(cudaMalloc((void**)&dev_P, sizeP));

    checkCuda(cudaMemcpy(dev_M, M, sizeM, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_N, N, sizeN, cudaMemcpyHostToDevice));

    // Configure grid and block sizes
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 blocksPerGrid((l + TILE_WIDTH - 1) / TILE_WIDTH, (j + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    // Launch kernels 
    int mul_gpu = SHARED;
    if (mul_gpu == NAIVE) {
        matMulKernelNaive << <blocksPerGrid, threadsPerBlock >> > (dev_M, dev_N, dev_P, j, k, l);
    }
    else if (mul_gpu == SHARED){
        matMulKernelShared << <blocksPerGrid, threadsPerBlock >> > (dev_M, dev_N, dev_P, j, k, l);
    }

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());

    checkCuda(cudaDeviceSynchronize());

    // Copy result back to host
    checkCuda(cudaMemcpy(P, dev_P, sizeP, cudaMemcpyDeviceToHost));

    // Free GPU memory
    checkCuda(cudaFree(dev_M));
    checkCuda(cudaFree(dev_N));
    checkCuda(cudaFree(dev_P));
}

int main()
{
    int j = 2000, k = 2500, l = 3000;  // Experiment with 100x100, 1000x1000,(2000,2500,3000).
    size_t sizeM = j * k * sizeof(float);
    size_t sizeN = k * l * sizeof(float);
    size_t sizeP = j * l * sizeof(float);

    // Allocate host matrices
    float* M = (float*)malloc(sizeM);
    float* N = (float*)malloc(sizeN);
    float* P_cpu = (float*)malloc(sizeP);
    float* P_gpu = (float*)malloc(sizeP);

    // Initialize matrices
    srand((unsigned int)time(NULL));
    for (int i = 0; i < j * k; ++i) M[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < k * l; ++i) N[i] = (float)rand() / RAND_MAX;

    // CPU matrix multiplication
    matMulCPU(M, N, P_cpu, j, k, l);

    // GPU matrix multiplication
    matMulGPU(M, N, P_gpu, j, k, l);

    // Verify results
    verifyResults(P_cpu, P_gpu, j * l);

    // Free host memory
    free(M);
    free(N);
    free(P_cpu);
    free(P_gpu);

    return 0;
}


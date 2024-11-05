#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void noDivergence(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void singleBranch(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    if (threadIdx.x % 32 < 16) {
        c[i] = a[i] + b[i];
    }
    else {
        c[i] = a[i] * b[i];
    }
}

__global__ void nestedBranch(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    if (threadIdx.x % 32 < 8) {
        c[i] = a[i] + b[i];
    }
    else if (threadIdx.x % 32 < 8){
        c[i] = a[i] + 2 * b[i];
    }
    else if (threadIdx.x % 32 < 16) {
        c[i] = a[i] + 3 * b[i];
    }
    else if (threadIdx.x % 32 < 24) {
        c[i] = a[i] + 4 * b[i];
    }
}

int main()
{
    const int arraySize = 32 * 1e5;
    int* a = (int*)malloc(arraySize * sizeof(int));
    int* b = (int*)malloc(arraySize * sizeof(int));
    int* c = (int*)malloc(arraySize * sizeof(int));

    for (int x = 0, int* pa = a, int* pb = b; x < arraySize; x++)
    {
        *pa++ = x + 1;
        *pb++ = 2 * (x + 1);
    }

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threadPerBlock(32, 1, 1);
    dim3 blockPerGrid((size + threadPerBlock.x - 1) / threadPerBlock.x, 1, 1);
    // Launch a kernel on the GPU with one thread for each element.
    noDivergence << <blockPerGrid, threadPerBlock >> > (dev_c, dev_a, dev_b);
    cudaStatus = cudaDeviceSynchronize();
    singleBranch << <blockPerGrid, threadPerBlock >> > (dev_c, dev_a, dev_b);
    cudaStatus = cudaDeviceSynchronize();
    nestedBranch << <blockPerGrid, threadPerBlock >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

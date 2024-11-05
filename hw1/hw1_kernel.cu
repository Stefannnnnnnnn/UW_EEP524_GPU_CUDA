#include "device_launch_parameters.h"
#include "check_error.h"
#include <cmath>

void printfGPU(void);
void SAXPYGPU(const float* x, const float a, const int N, float* y);
void _2DAddGPU(const float* A, const float* B, const int M, const int N, float* C);
void _3DAddGPU(const float* A, const float* B, const int M, const int N, const int P, float* C);

__global__ void printfKernel(void)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    printf("thread ID: %d, blockID: %d, GTID: %d\n", threadIdx.x, blockIdx.x, gid);
}

__global__ void SAXPYKernel(const float* x, const float a, const int N, float* y)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < N) {
        y[gid] = a * x[gid] + y[gid];
    }
}

__global__ void _2DAddKernel(const float* A, const float* B, const int M, const int N, float* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int index = row * N + col;
        C[index] = A[index] + B[index];
    }
}

__global__ void _3DAddKernel(const float* A, const float* B, const int M, const int N, const int P, float* C)
{
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (depth < M && row < N && col < P) {
        int index = depth * N * P + row * P + col;
        C[index] = A[index] + B[index];
    }
}

// CPU version of Single-Precision A times X plus Y
void SAXPYCPU(const float* x, const float a, const int N, float* y)
{
    for (int i = 0; i < N; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// CPU version of 2D matrix addition
void _2DAddCPU(const float* A, const float* B, const int M, const int N, float* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
}
// CPU version of 3D grid addition
void _3DAddCPU(const float* A, const float* B, const int M, const int N, const int P, float* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                C[i * N * P + j * P + k] 
              = A[i * N * P + j * P + k] 
              + B[i * N * P + j * P + k];
            }
        }
    }
}

int main()
{
    float fTol = 1e-100;
    
    // Test printf kernel
    if (1) {
        printfGPU();
    }

    // Test SAXPY kernel
    if (0) {
        const int N = 1e7;
        const float a = 3.5;
        float* x_gpu = (float*)malloc(N * sizeof(float));
        float* y_gpu = (float*)malloc(N * sizeof(float));
        float* x_cpu = (float*)malloc(N * sizeof(float));
        float* y_cpu = (float*)malloc(N * sizeof(float));
        bool is_correct = true;

        for (int i = 0; i < N; i++) {
            x_gpu[i] = 1.5 * i;
            x_cpu[i] = 1.5 * i;
            y_gpu[i] = 2.5 * i;
            y_cpu[i] = 2.5 * i;
        }
        SAXPYGPU(x_gpu, a, N, y_gpu);
        SAXPYCPU(x_cpu, a, N, y_cpu);

        // Compare CPU&GPU results 
        for (int i = 0; i < N; i++) {
            if (abs(y_gpu[i] - y_cpu[i]) > fTol) {
                is_correct = false;
                break;
            }
        }
        is_correct ? printf("SAXPYKernel worked!\n") : printf("SAXPYKernel failed!\n");

        free(x_gpu);
        free(y_gpu);
        free(x_cpu);
        free(y_cpu);
    }

    // Test matrix add kernel
    if (0) {
        const int M = 3024;
        const int N = 4032;
        float* A = (float*)malloc(M * N * sizeof(float));
        float* B = (float*)malloc(M * N * sizeof(float));
        float* C_gpu = (float*)malloc(M * N * sizeof(float));
        float* C_cpu = (float*)malloc(M * N * sizeof(float));
        bool is_correct = true;

        for (int i = 0; i < M * N; i++) {
            A[i] = i;
            B[i] = 2.05 * i;
        }
        _2DAddGPU(A, B, M, N, C_gpu);
        _2DAddCPU(A, B, M, N, C_cpu);

        // Compare CPU&GPU results 
        for (int i = 0; i < M * N; i++) {
            if (abs(C_gpu[i] - C_cpu[i]) > fTol) {
                is_correct = false;
                break;
            }
        }
        is_correct ? printf("2DAddKernel worked!\n") : printf("2DAddKernel failed!\n");

        free(A);
        free(B);
        free(C_gpu);
        free(C_cpu);
    }

    // Test grid add kernel
    if (0) {
        const int M = 100;
        const int N = 100;
        const int P = 100;
        float* A = (float*)malloc(M * N * P * sizeof(float));
        float* B = (float*)malloc(M * N * P * sizeof(float));
        float* C_gpu = (float*)malloc(M * N * P * sizeof(float));
        float* C_cpu = (float*)malloc(M * N * P * sizeof(float));
        bool is_correct = true;

        for (int i = 0; i < M * N * P; i++) {
            A[i] = i;
            B[i] = 2.05 * i;
        }
        _3DAddGPU(A, B, M, N, P, C_gpu);
        _3DAddCPU(A, B, M, N, P, C_cpu);

        // Compare CPU&GPU results 
        for (int i = 0; i < M * N * P; i++) {
            if (abs(C_gpu[i] - C_cpu[i]) > fTol) {
                is_correct = false;
                break;
            }
        }
        is_correct ? printf("3DAddKernel worked!\n") : printf("3DAddKernel failed!\n");

        free(A);
        free(B);
        free(C_gpu);
        free(C_cpu);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());

    return 0;
}

void printfGPU(void) {
    int threadsPerBlock = 1;
    int blocksPerGrid = 32;
    printfKernel<<<blocksPerGrid, threadsPerBlock>>>();
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

void SAXPYGPU(const float* x, const float a, const int N, float* y) {
    float* dev_x = 0;
    float* dev_y = 0;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers for two vectors (one input, one outin).
    checkCuda(cudaMalloc((void**)&dev_x, N * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_y, N * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the SAXPYKernel
    SAXPYKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_x, a, N, dev_y);
    checkCuda(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(y, dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(dev_x);
    cudaFree(dev_y);
}

void _2DAddGPU(const float* A, const float* B, const int M, const int N, float* C) {
    float* dev_A = 0;
    float* dev_B = 0;
    float* dev_C = 0;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers for vectors
    checkCuda(cudaMalloc((void**)&dev_A, M * N * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_B, M * N * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_C, M * N * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_B, B, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the _2dAddKernel
    _2DAddKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, M, N, dev_C);
    checkCuda(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

void _3DAddGPU(const float* A, const float* B, const int M, const int N, const int P, float* C) {
    float* dev_A = 0;
    float* dev_B = 0;
    float* dev_C = 0;
    dim3 threadsPerBlock{8, 8, 8};
    dim3 blocksPerGrid((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y, 
        (M + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers for vectors 
    checkCuda(cudaMalloc((void**)&dev_A, M * N * P * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_B, M * N * P * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_C, M * N * P * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_A, A, M * N * P * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_B, B, M * N * P * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the _3dAddKernel
    _3DAddKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, M, N, P, dev_C);
    checkCuda(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(C, dev_C, M * N * P * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}
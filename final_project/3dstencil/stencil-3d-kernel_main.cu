//==============================================================================
// stencil-3d-kernel-REL.cu
//
// last modified: 11/3/2024
// status: 
// outputs: 
// performs host-side verification result checks for all 3 kernel outputs 

//==== NOTES : 3D stencil kernel variants
// contains kernels:
// - stencil_3d_basic_kernel
// - stencil_3d_thread_coarsening_kernel
// - stencil_3d_tiled_smem_kernel
//
//==============================================================================
//#include "profileapi.h"

#include "stencil_3d_kernels.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "windows.h" // include for QueryPerfTimer API instead of profile.h

using namespace std;

#define BASIC 0
#define SQR 1
#define NONSQR 2
#define COARSEN 3

// for Win API QPC host timing
LARGE_INTEGER StartingTime, StartingTime2, EndingTime, ElapsedMicroseconds;
LARGE_INTEGER Frequency;

const int N = 128; // width of cubic 3D grid in each dimension

// #include your error-check macro header file here
// Requires Error: section at end of Main function
inline static cudaError_t cudartCHK(cudaError_t result, char* srcStr)                                                 
{                                                                       
    if (result != cudaSuccess)                                            
    {                                                                     
        printf("CUDA Runtime API Error (src: %s): %s:%d\n", srcStr, __FILE__, __LINE__);                      
        printf("error code:%d, name: %s reason: %s\n", result,cudaGetErrorName(result), cudaGetErrorString(result));
    }
    return result;
}

// TODO: move this into UTIL header file and share between all user files (stencil-3d, conv-2d)
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

        cout << "Entire file content of file: " << strFile << " is in memory of size: " << size << " bytes." << endl;
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


void stencil_3d_basic_host(float* d_IN, float* d_OUT, float* h_OUT1, int bytes_read) {
    // each dim (x,y,z) should have total # threads = corresponding input 3D grid dim
    // e.g. grid_dim.x = data_dim.x, grid_dim.y = data_dim.y, grid_dim.z = data_dim.z
    dim3 dimBlock3D(32, 4, 4);  // optimized for coalescing warps
    //dim3 dimGrid3D(24, 192, 192); // for 768^3 data dims
    //dim3 dimGrid3D(16, 128, 128); // for 512x512x512 data dims
    dim3 dimGrid3D(4, 32, 32); // for 128^3 data dims
    //dim3 dimGrid3D(1, 4, 8); // for 32^3 data dims

    cudartCHK(cudaMemset(d_OUT, 0, bytes_read), "cudaMemset-1");
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-1");
    stencil_3d_basic_kernel << <dimGrid3D, dimBlock3D >> > (d_IN, d_OUT, N);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-2");
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError(), "cudaGetLastError-1");
    // retrieve result data from device back to host
    // Copy output image from GPU buffer to host memory.
    cudartCHK(cudaMemcpy(h_OUT1, d_OUT, bytes_read, cudaMemcpyDeviceToHost), "cudaMemcpy-2");
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-3");
}

void stencil_3d_tiled_smem_host(float* d_IN, float* d_OUT, float* h_OUT2, int bytes_read) {
    dim3 dimBlock3D(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid3D(ceil(float(N) / float(OUT_TILE_DIM)), // needs to cover output grid dims using OUT_TILE_DIM size!
                   ceil(float(N) / float(OUT_TILE_DIM)),
                   ceil(float(N) / float(OUT_TILE_DIM)));
    cudartCHK(cudaMemset(d_OUT, 0, bytes_read), "cudaMemset-2");
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-4");
    stencil_3d_tiled_smem_kernel<< <dimGrid3D, dimBlock3D >> > (d_IN, d_OUT, N);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-5");
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError(), "cudaGetLastError-2");

    // retrieve result data from device back to host
    // Copy output image from GPU buffer to host memory.
    cudartCHK(cudaMemcpy(h_OUT2, d_OUT, bytes_read, cudaMemcpyDeviceToHost), "cudaMemcpy-3");
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-6");
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError(), "cudaGetLastError-3");
}

void stencil_3d_nonsqr_tiled_smem_host(float* d_IN, float* d_OUT, float* h_OUT2, int bytes_read) {
    dim3 dimBlock3D(IN_TILE_DIM_X, IN_TILE_DIM_YZ, IN_TILE_DIM_YZ);
    dim3 dimGrid3D(ceil(float(N) / float(OUT_TILE_DIM_X)), // needs to cover output grid dims using OUT_TILE_DIM size!
                   ceil(float(N) / float(OUT_TILE_DIM_YZ)), ceil(float(N) / float(OUT_TILE_DIM_YZ)));
    cudartCHK(cudaMemset(d_OUT, 0, bytes_read), "cudaMemset-2");
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-4");
    stencil_3d_nonsqr_tiled_smem_kernel << <dimGrid3D, dimBlock3D >> > (d_IN, d_OUT, N);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-5");
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError(), "cudaGetLastError-2");

    // retrieve result data from device back to host
    // Copy output image from GPU buffer to host memory.
    cudartCHK(cudaMemcpy(h_OUT2, d_OUT, bytes_read, cudaMemcpyDeviceToHost), "cudaMemcpy-3");
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-6");
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError(), "cudaGetLastError-3");
}

void stencil_3d_thread_coarsening_host(float* d_IN, float* d_OUT, float* h_OUT3, int bytes_read) {
    dim3 dimBlock3D(IN_TILE_DIMB, IN_TILE_DIMB, 1); // only 1 thread in Z dim!
    dim3 dimGrid3D(ceil(float(N) / float(OUT_TILE_DIMB)), // assuming N evenly divisible by IN_TILE_DIMB (32) for now!
                   ceil(float(N) / float(OUT_TILE_DIMB)),
                   ceil(float(N) / float(OUT_TILE_DIMB)));

    cudartCHK(cudaMemset(d_OUT, 0, bytes_read), "cudaMemset-3");
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-7");
    stencil_3d_thread_coarsening_kernel << <dimGrid3D, dimBlock3D >> > (d_IN, d_OUT, N);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-8");
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError(), "cudaGetLastError-4");
    // retrieve result data from device back to host
    // Copy output image from GPU buffer to host memory.
    cudartCHK(cudaMemcpy(h_OUT3, d_OUT, bytes_read, cudaMemcpyDeviceToHost), "cudaMemcpy-4");
    cudartCHK(cudaDeviceSynchronize(), "cudaDeviceSynchronize-9");
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError(), "cudaGetLastError-5");
    // STOP timer #1
}


int main(int argc, char** argv)
{
    //load 3D float data grid - assumed NxNxN cube
    ostringstream oss;
    oss << "../dat/3dgrid_" << N << "x" << N << "x" << N << "_float_assorted.bin";
    string strFile = oss.str();

    char* memblk = NULL;
    int bytes_read = read_binary_grid_file(strFile, &memblk);
    float* h_IN = (float*)memblk;

    printf("<stencil-3d-kernel-REL.cu> main: reading 3D data input: %s\n", strFile.c_str());
    printf("<stencil-3d-kernel-REL.cu> main: read %d bytes of 3D data input grid. Assuming %dx%dx%d grid dims!\n", bytes_read, N, N, N);

    int numFloats = bytes_read / sizeof(float);

    //// setup host variables, allocate host memory as needed
    //unsigned char* h_imgOut = (unsigned char*)malloc(imgSize);
    float* h_OUT1 = (float*)malloc(bytes_read);
    float* h_OUT2 = (float*)malloc(bytes_read);
    float* h_OUT3 = (float*)malloc(bytes_read);
    float* h_OUT4 = (float*)malloc(bytes_read);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudartCHK(cudaSetDevice(0),"cudaSetDevice");

    // allocate device memory
    float* d_IN = 0;
    float* d_OUT = 0;

    // START timer #1
    // TIME full host-side round trip execution time including GPU mem alloc & transfers to/from
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);

    cudartCHK(cudaMalloc((void**)&d_IN, bytes_read),"cudaMalloc d_IN");
    cudartCHK(cudaMalloc((void**)&d_OUT, bytes_read), "cudaMalloc d_OUT");

    // Copy input data from host memory to GPU buffers.
    cudartCHK(cudaMemcpy(d_IN, h_IN, bytes_read, cudaMemcpyHostToDevice),"cudaMemcpy-1");

    const int kernel_type = COARSEN;
    // ====== KERNEL 1
    if (kernel_type == BASIC) stencil_3d_basic_host(d_IN, d_OUT, h_OUT1, bytes_read);
    // ====== KERNEL 2 : SMEM Tiled
    else if (kernel_type == SQR) stencil_3d_tiled_smem_host(d_IN, d_OUT, h_OUT2, bytes_read);
    // ====== KERNEL 3 : SMEM Tiled nonsqr
    else if (kernel_type == NONSQR) stencil_3d_nonsqr_tiled_smem_host(d_IN, d_OUT, h_OUT3, bytes_read);
    // ====== KERNEL 4 : coarsen
    else if (kernel_type == COARSEN) stencil_3d_thread_coarsening_host(d_IN, d_OUT, h_OUT4, bytes_read);

    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    // report Kernel-1 host-side timing
    if (kernel_type == BASIC) cout << "Kernel-1 (naive GMEM) 3D stencil sweep host round-trip time: " 
                                   << ElapsedMicroseconds.QuadPart << " (us)" << endl;
    else if (kernel_type == SQR) cout << "Kernel-2 (Tile SMEM) 3D stencil sweep host round-trip time: "
                                         << ElapsedMicroseconds.QuadPart << " (us)" << endl;
    else if (kernel_type == NONSQR) cout << "Kernel-3 (nonsqr SMEM) 3D stencil sweep host round-trip time: "
                                         << ElapsedMicroseconds.QuadPart << " (us)" << endl;
    else if (kernel_type == COARSEN) cout << "Kernel-4 (coarsen) 3D stencil sweep host round-trip time: "
                                          << ElapsedMicroseconds.QuadPart << " (us)" << endl;
    cudartCHK(cudaDeviceReset(), "cudaDeviceReset"); // must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    
    //=====================================================================================================
    // =========== verify results
    // compute host verification routine
    // compute output element by applying 3D stencil coefficients to input elements

    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);

    float* h_CPUREF = (float*)malloc(numFloats * sizeof(float));
    memset(h_CPUREF, 0, numFloats * sizeof(float));
    // omitting 1st and last values in each dim, in same was GPU kernels do, to avoid boundary condition issues
    QueryPerformanceCounter(&StartingTime2);
    for (int i = 1; i < N-1; ++i) // Z-slice dim!
    {
        for (int j = 1; j < N-1; ++j) // Y-rows dim!
        {
            for (int k = 1; k < N-1; ++k) // X-cols dim = fastest moving!
            {
                h_CPUREF[i*N*N + j*N + k] = h_7pt_stencil_coeffs[0]* h_IN[i * N * N + j * N + k]
                    + h_7pt_stencil_coeffs[1] * h_IN[i * N * N + j * N + (k - 1)]
                    + h_7pt_stencil_coeffs[2] * h_IN[i * N * N + j * N + (k + 1)]
                    + h_7pt_stencil_coeffs[3] * h_IN[i * N * N + (j - 1) * N + k]
                    + h_7pt_stencil_coeffs[4] * h_IN[i * N * N + (j + 1) * N + k]
                    + h_7pt_stencil_coeffs[5] * h_IN[(i - 1) * N * N + j * N + k]
                    + h_7pt_stencil_coeffs[6] * h_IN[(i + 1) * N * N + j * N + k];
            }
        }
    }
    QueryPerformanceCounter(&EndingTime);

    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    cout << "CPU Reference 3D stencil sweep sequential routine time: " << ElapsedMicroseconds.QuadPart << " (us)" << endl;

    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime2.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    cout << "CPU Reference 3D stencil kernel time: " << ElapsedMicroseconds.QuadPart << " (us)" << endl;
    // retrieve and save timer results (write to console or file)
    //=====================================================================================================

    float ERR_TOL = 2e-4;
    float cpuref_maxerr = 0;
    float gmemcoarse_maxerr = 0;
    int cpuref_errCnt = 0; // CPU Reference v GMEM basic GPU kernel - should be baseline reference verification
    int posValCnt = 0;
    int cpuRefZeroCnt = 0;
    int i = 0;

    float* h_OUT_;
    if (kernel_type == BASIC) h_OUT_ = h_OUT1;
    else if (kernel_type == SQR) h_OUT_ = h_OUT2;
    else if (kernel_type == NONSQR) h_OUT_ = h_OUT3;
    else h_OUT_ = h_OUT4;

    for (i = 0; i < numFloats; ++i)
    {
        if (abs(h_CPUREF[i] - h_OUT_[i]) > ERR_TOL)
        {
            cpuref_errCnt++;
            float err = abs(h_CPUREF[i] - h_OUT_[i]);
            if (err > cpuref_maxerr)
                cpuref_maxerr = err;
        }
        if (h_OUT_[i] > 0.0f)
            posValCnt++;
        if (h_CPUREF[i] == 0.0f)
            cpuRefZeroCnt++;
    }
    cout << "VERIFICATION: # output grid elements = " << numFloats << endl;
    cout << "VERIFICATION: error tolerance threshold = " << ERR_TOL << endl;
    cout << "VERIFICATION: total # output values checked = " << i << endl;
    cout << "VERIFICATION: total # errors between CPU-REF & GMEM stencil results = " << cpuref_errCnt << endl;
    cout << "VERIFICATION: max error between CPU-REF & GMEM  = " << cpuref_maxerr << endl;

    cout << "VERIFICATION: total non-zero agreed results in output grid  = " << posValCnt << endl;
    cout << "VERIFICATION: CPU REF identically-zero count  = " << cpuRefZeroCnt << endl;

//Error:  // assumes error macro has a goto Error statement

    // free host and device memory
    delete[] memblk;
    free(h_OUT1);
    free(h_OUT2);
    free(h_OUT3);
    free(h_OUT4);
    cudaFree(d_IN);
    cudaFree(d_OUT);

    return 0;
}




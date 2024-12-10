//==============================================================================
// parhist_main.cu : parallel histogram CUDA
//
// last modified: 11/12/2024
// status: 
// outputs: 
// performs host-side verification result checks for all CUDA kernel outputs 
// - histo_seq_host_cpu( )

//==== NOTES : parallel histogram kernels
// contains kernels:
// - histo_basic_kernel [v0]
// - histo_priv_global_kernel [v1]
// - histo_priv_smem_kernel [v2]
// - histo_coarsen_contiguous_kernel [v3]
// - histo_coarsen_interleaved_kernel [v4]
// 
//==============================================================================
//#include "profileapi.h"

#include "parallel_histogram_kernels.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "windows.h" // include for QueryPerfTimer API instead of profile.h
using namespace std;

// for Win API QPC host timing
LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
LARGE_INTEGER Frequency;

//#define STB_IMAGE_IMPLEMENTATION
//#include "../util/stb_image.h"  // download from class website files
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "../util/stb_image_write.h"  // download from class website files

// #include your error-check macro header file here

// #include your error-check macro header file here
cudaError_t cudartCHK(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        result = cudaGetLastError(); // pull last error off error stack and reset to cudaSuccess
        //printf("CUDA Runtime API Error: %s:%d\n", __FILE__, __LINE__);
        printf("error code:%d, name: %s reason: %s\n", result, cudaGetErrorName(result), cudaGetErrorString(result));
    }
    return result;
}
//#define cudartCHK(call)                                                  \
//{                                                                        \
//   const cudaError_t error = call;                                       \
//   if (error != cudaSuccess)                                             \
//   {                                                                     \
//      printf("CUDA Runtime API Error: %s:%d\n", __FILE__, __LINE__);                      \
//      printf("error code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
//      goto Error;                                                           \
//   }                                                                     \
//}


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

int main(int argc, char** argv)
{

    //string strOutfile = "C:/CODE/AUT24_GPUCompute/dat/random_alphabetics_utf8_n10M.bin";

    //load ASCII 8-bit character data
    //string strInFile = "C:/CODE/AUT24_GPUCompute/dat/random_lowercase_utf8_n10M.txt"; 

    string strInFile;
    strInFile = "C:/AUT24_GPUComp/Part2/Part2/random_lowercase_utf8_n1M.txt";
    /*
    if (argc == 2)
    {
        strInFile = argv[1];
    }
    else
    {
        cout << "INPUT ERROR: parhisto2.exe TAKES 1 input argument: <full path to input char array file>!" << endl;
        exit(-1);
    }
    */

    char* memblk = NULL;
    int charDataBytes = read_binary_grid_file(strInFile, &memblk);
    char* h_IN = (char*)memblk;

    int numHistoBins = 26; // currently using 1-character bins for lowercase alphabetic

    //// setup host variables, allocate host memory as needed
    uint* h_OUT1 = (uint*)malloc(numHistoBins * sizeof(uint));
    memset(h_OUT1, 0, numHistoBins * sizeof(uint));
    uint* h_OUT2 = (uint*)malloc(numHistoBins * sizeof(uint));
    memset(h_OUT2, 0, numHistoBins * sizeof(uint));
    uint* h_OUT3 = (uint*)malloc(numHistoBins * sizeof(uint));
    memset(h_OUT3, 0, numHistoBins * sizeof(uint));
    uint* h_OUT4 = (uint*)malloc(numHistoBins * sizeof(uint));
    memset(h_OUT4, 0, numHistoBins * sizeof(uint));
    uint* h_OUT5 = (uint*)malloc(numHistoBins * sizeof(uint));
    memset(h_OUT5, 0, numHistoBins * sizeof(uint));
    uint* h_OUT6 = (uint*)malloc(numHistoBins * sizeof(uint));
    memset(h_OUT6, 0, numHistoBins * sizeof(uint));

    // =========== verify results
    // compute CPU host verification routine
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);

    // run CPU sequential histogram reference
    histo_seq_host_cpu(h_IN, charDataBytes, h_OUT1);

    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

    // We now have the elapsed number of ticks, along with the number of ticks-per-second. We use these values
    // to convert to the number of elapsed microseconds. To guard against loss-of-precision, we convert
    // to microseconds *before* dividing by ticks-per-second.
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

    // report CPU Ref timing
    cout << "CPU Reference histogram sequential routine time: " << ElapsedMicroseconds.QuadPart << " (us)" << endl;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudartCHK(cudaSetDevice(0));

    // START timer #1

    // allocate device memory
    char* d_IN = 0;
    uint* d_OUT = 0;

    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);
    cout << "starting kernel #1. numHistoBins = " << numHistoBins << endl;
    cudartCHK(cudaMalloc((void**)&d_IN, charDataBytes));
    cudartCHK(cudaMalloc((void**)&d_OUT, numHistoBins*sizeof(uint)));

    // Copy input data from host memory to GPU buffers.
    cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice));

    // ==== BASIC NAIVE v0 GPU histogram kernel
    // Launch a 1D kernel on the GPU with specified launch configuration dimensions
    dim3 dimBlock2D(256, 1, 1);
    dim3 dimGrid2D((charDataBytes + dimBlock2D.x - 1) / dimBlock2D.x, 1, 1);

    cudartCHK(cudaMemset(d_OUT, 0, numHistoBins*sizeof(uint)));
    cudartCHK(cudaDeviceSynchronize());

    histo_basic_kernel << <dimGrid2D, dimBlock2D >> > (d_IN, charDataBytes, d_OUT);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudartCHK(cudaDeviceSynchronize());
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError());
    // retrieve result data from device back to host
    // Copy output image from GPU buffer to host memory.
    cudartCHK(cudaMemcpy(h_OUT2, d_OUT, numHistoBins * sizeof(uint), cudaMemcpyDeviceToHost));
    cudartCHK(cudaDeviceSynchronize());

    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    cout << "histo_basic_kernel routine time: " << ElapsedMicroseconds.QuadPart << " (us)" << endl;

 
    // ==== PRIVATIZATION - Global/L2
    // QueryPerformanceFrequency(&Frequency);
    // QueryPerformanceCounter(&StartingTime);
     //cout << "starting kernel #2" << endl;
     // cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice)); //
   //cudartCHK(cudaMemset(d_OUT, 0, numHistoBins * sizeof(uint)));
    //cout << "CHKPT-1" << endl;
    //cudartCHK(cudaDeviceSynchronize());
    //cout << "CHKPT-2" << endl;
    //histo_priv_global_kernel << <dimGrid2D, dimBlock2D >> > (d_IN, charDataBytes, numHistoBins, d_OUT);
    //cudartCHK(cudaDeviceSynchronize());
    //cout << "CHKPT-3" << endl;
    //// Check for any errors launching the kernel
    //cudartCHK(cudaGetLastError());
    //cout << "CHKPT-4" << endl;
    //// retrieve result data from device back to host
    //// Copy output image from GPU buffer to host memory.
    //cudartCHK(cudaMemcpy(h_OUT3, d_OUT, numHistoBins * sizeof(uint), cudaMemcpyDeviceToHost));
    //cout << "CHKPT-5" << endl;
    //cudartCHK(cudaDeviceSynchronize());
    
    //QueryPerformanceCounter(&EndingTime);
    //ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    //ElapsedMicroseconds.QuadPart *= 1000000;
    //ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    //cout << "histo_priv_global_kernel routine time: " << ElapsedMicroseconds.QuadPart << " (us)" << endl;

    // ==== PRIVATIZATION - SMEM
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);
    cout << "starting kernel #3" << endl;
    cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice)); //
    cudartCHK(cudaMemset(d_OUT, 0, numHistoBins * sizeof(uint)));
    cudartCHK(cudaDeviceSynchronize());
    histo_priv_smem_kernel << <dimGrid2D, dimBlock2D >> > (d_IN, charDataBytes, d_OUT);
    cudartCHK(cudaDeviceSynchronize());
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError());
    // retrieve result data from device back to host
    // Copy output image from GPU buffer to host memory.
    cudartCHK(cudaMemcpy(h_OUT4, d_OUT, numHistoBins * sizeof(uint), cudaMemcpyDeviceToHost));
    cudartCHK(cudaDeviceSynchronize());

    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    cout << "histo_priv_smem_kernel routine time : " << ElapsedMicroseconds.QuadPart << " (us)" << endl;

    // ==== COARSENED - CONTIGUOUS
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);
    cout << "starting kernel #4" << endl;
    cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice)); //
    cudartCHK(cudaMemset(d_OUT, 0, numHistoBins * sizeof(uint)));
    cudartCHK(cudaDeviceSynchronize());

    uint CFACT = 16;
    dimGrid2D.x = ceil(float(dimGrid2D.x) / float(CFACT));  // reduce grid size to let each coarsened thread do more work

    histo_coarsen_contiguous_kernel<< <dimGrid2D, dimBlock2D >> > (d_IN, charDataBytes, d_OUT, CFACT);
    cudartCHK(cudaDeviceSynchronize());
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError());
    // retrieve result data from device back to host
    // Copy output image from GPU buffer to host memory.
    cudartCHK(cudaMemcpy(h_OUT6, d_OUT, numHistoBins * sizeof(uint), cudaMemcpyDeviceToHost));
    cudartCHK(cudaDeviceSynchronize());

    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    cout << "histo_coarsen_contiguous_kernel routine time : " << ElapsedMicroseconds.QuadPart << " (us)" << endl;

    // ==== COARSENED - INTERLEAVED
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);
    cout << "starting kernel #5" << endl;
    cudartCHK(cudaMemcpy(d_IN, h_IN, charDataBytes, cudaMemcpyHostToDevice)); //
    cudartCHK(cudaMemset(d_OUT, 0, numHistoBins * sizeof(uint)));
    cudartCHK(cudaDeviceSynchronize());

    histo_coarsen_interleaved_kernel << <dimGrid2D, dimBlock2D >> > (d_IN, charDataBytes, d_OUT);
    cudartCHK(cudaDeviceSynchronize());
    // Check for any errors launching the kernel
    cudartCHK(cudaGetLastError());
    // retrieve result data from device back to host
    // Copy output image from GPU buffer to host memory.
    cudartCHK(cudaMemcpy(h_OUT5, d_OUT, numHistoBins * sizeof(uint), cudaMemcpyDeviceToHost));
    cudartCHK(cudaDeviceSynchronize());

    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    cout << "histo_coarsen_interleaved_kernel routine time : " << ElapsedMicroseconds.QuadPart << " (us)" << endl;
   
    cudartCHK(cudaDeviceReset()); // must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    cout << "All kernels complete." << endl;
    // verify results
    int errCnt1 = 0;  // CPU v GPU basic
    int errCnt2 = 0; // 
    int errCnt3 = 0; // 
    int errCnt4 = 0; //
    int errCnt5 = 0; //
    int errCnt6 = 0;
    int posValCnt = 0;
    int totalCharCnt = 0; // count of total alphabetic characters binned
    int i = 0;
    for (i = 0; i < numHistoBins; ++i)
    {
        if (h_OUT1[i] != h_OUT2[i])
        {
            errCnt1++;
        }
        //if (h_OUT1[i] != h_OUT3[i])
        //{
        //    errCnt2++;
        //}
        //if (h_OUT2[i] != h_OUT3[i])
        //{
        //    errCnt3++;
        //}
        if (h_OUT1[i] != h_OUT4[i])
        {
            errCnt4++;
        }
        if (h_OUT1[i] != h_OUT5[i])
        {
            errCnt5++;
        }
        if (h_OUT1[i] != h_OUT6[i])
        {
            errCnt6++;
        }

        totalCharCnt += h_OUT1[i];

        cout << "histo[" << i << "] = " << h_OUT1[i] << endl;

    //    if (h_OUT1[i] > 0.0f)
    //        posValCnt++;
    }
    cout << "VERIFICATION: # output histogram bins = " << numHistoBins << endl;
    cout << "VERIFICATION: total # output values checked = " << i << endl;
    cout << "VERIFICATION: total # errors between CPU and Basic_Histo results = " << errCnt1 << endl;
    //cout << "VERIFICATION: total # errors between CPU & Priv_Global_Histo results = " << errCnt2 << endl;
    //cout << "VERIFICATION: total # errors between Basic_Hist and Priv_Global_Histo results = " << errCnt3 << endl;
    cout << "VERIFICATION: total # errors between CPU and Priv_SMEM_Histo results = " << errCnt4 << endl;
    cout << "VERIFICATION: total # errors between CPU and Coars_Intlv_Histo results = " << errCnt5 << endl;
    cout << "VERIFICATION: total # errors between CPU and Coars_Contig_Histo results = " << errCnt6 << endl;
    cout << "VERIFICATION: total alphabetic characters binned = " << totalCharCnt << endl;

    //cout << "VERIFICATION: total non-zero agreed results in output grid  = " << posValCnt << endl;

    //write_bin_grid_file(strOutfile, (char*)h_OUT1, numHistoBins * sizeof(uint));

     // EXTRA CREDIT:
    // retrieve and save timer results (write to console or file)

//
    // free host and device memory
    free(h_OUT1);
    free(h_OUT2);
    free(h_OUT3);
    cudaFree(d_IN);
    cudaFree(d_OUT);

    return 0;
}




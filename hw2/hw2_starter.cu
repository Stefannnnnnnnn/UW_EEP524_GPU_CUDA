#include <windows.h>
#include "../util/profileapi.h"
#include <device_launch_parameters.h> 
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../util/stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../util/stb_image_write.h"  

#include "../util/check_error.h"

// global gaussian blur filter coefficients array here
#define BLUR_FILTER_WIDTH 9  // 9x9 (square) Gaussian blur filter
const float BLUR_FILT[81] = { 0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.3292,0.5353,0.7575,0.9329,1.0000,0.9329,0.7575,0.5353,0.3292,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084};

#define USE_GPU 1
#define GPU_GLOBAL 0
#define GPU_STATIC 1
#define GPU_DYNAMIC 2

// DEFINE your CUDA blur kernel function(s) here
// blur kernel #1 - global memory only
__global__ void blurKernelGlobal(const int w, const int h, const float* filter, const unsigned char* in, unsigned char* out)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) {
        float pixVal = 0;
        float filterSum = 0;
        const int _half_filter_width = BLUR_FILTER_WIDTH / 2;
        // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
        for (int blurRow = -_half_filter_width; blurRow < _half_filter_width + 1; ++blurRow) {
            for (int blurCol = -_half_filter_width; blurCol < _half_filter_width + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                // Verify we have a valid image pixel
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w + curCol] *
                        filter[(blurRow + _half_filter_width) * BLUR_FILTER_WIDTH + blurCol + _half_filter_width];
                    filterSum +=
                        filter[(blurRow + _half_filter_width) * BLUR_FILTER_WIDTH + blurCol + _half_filter_width];
                    // Keep track of number of pixels in the accumulated total
                }
            }
        }
        // Write our new pixel value out
        out[Row * w + Col] = (unsigned char)(pixVal / filterSum);
    }
}

// blur kernel #2 - device shared memory (static alloc)
__global__ void blurKernelSharedStatic(const int w, const int h, const float *filter, const unsigned char *in, unsigned char *out)
{
    __shared__ float static_filter[BLUR_FILTER_WIDTH][BLUR_FILTER_WIDTH];

    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // load filter into shared memory
    if (threadIdx.x < BLUR_FILTER_WIDTH && threadIdx.y < BLUR_FILTER_WIDTH) {
        static_filter[threadIdx.y][threadIdx.x] = filter[threadIdx.y * BLUR_FILTER_WIDTH + threadIdx.x];
    }
    __syncthreads();

    if (Col < w && Row < h) {
        float pixVal = 0;
        float filterSum = 0;
        const int _half_filter_width = BLUR_FILTER_WIDTH / 2;
        // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
        for (int blurRow = -_half_filter_width; blurRow < _half_filter_width + 1; ++blurRow) {
            for (int blurCol = -_half_filter_width; blurCol < _half_filter_width + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                // Verify we have a valid image pixel
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w + curCol] *
                        static_filter[(blurRow + _half_filter_width)][blurCol + _half_filter_width];
                    filterSum +=
                        static_filter[(blurRow + _half_filter_width)][blurCol + _half_filter_width];
                    // Keep track of number of pixels in the accumulated total
                }
            }
        }
        // Write our new pixel value out
        out[Row * w + Col] = (unsigned char)(pixVal / filterSum);
    }
}

extern __shared__ float dynamic_filter[];

// blur kernel #2 - device shared memory (dynamic alloc)
__global__ void blurKernelSharedDynamic(const int w, const int h, const float* filter, const unsigned char* in, unsigned char* out)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // load filter into shared memory
    if (threadIdx.x < BLUR_FILTER_WIDTH && threadIdx.y < BLUR_FILTER_WIDTH) {
        dynamic_filter[threadIdx.y * BLUR_FILTER_WIDTH + threadIdx.x] = filter[threadIdx.y * BLUR_FILTER_WIDTH + threadIdx.x];
    }
    __syncthreads();

    if (Col < w && Row < h) {
        float pixVal = 0;
        float filterSum = 0;
        const int _half_filter_width = BLUR_FILTER_WIDTH / 2;
        // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
        for (int blurRow = -_half_filter_width; blurRow < _half_filter_width + 1; ++blurRow) {
            for (int blurCol = -_half_filter_width; blurCol < _half_filter_width + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                // Verify we have a valid image pixel
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w + curCol] *
                        dynamic_filter[(blurRow + _half_filter_width) * BLUR_FILTER_WIDTH + blurCol + _half_filter_width];
                    filterSum +=
                        dynamic_filter[(blurRow + _half_filter_width) * BLUR_FILTER_WIDTH + blurCol + _half_filter_width];
                    // Keep track of number of pixels in the accumulated total
                }
            }
        }
        // Write our new pixel value out
        out[Row * w + Col] = (unsigned char)(pixVal / filterSum);
    }
}

// EXTRA CREDIT
// define host sequential blur-kernel routine
void cpu_blur(unsigned char* inputImg, unsigned char* outputImg, int imgWidth, int imgHeight) {
    int filterHalf = BLUR_FILTER_WIDTH / 2;

    // Iterate over every pixel in the image
    for (int y = 0; y < imgHeight; y++) {
        for (int x = 0; x < imgWidth; x++) {
            float pixelSum = 0;
            float filterSum = 0;

            // Apply the filter
            for (int ky = -filterHalf; ky <= filterHalf; ky++) {
                for (int kx = -filterHalf; kx <= filterHalf; kx++) {
                    // Verify we have a valid image pixel
                    if (y + ky > -1 && y + ky < imgHeight && x + kx > -1 && x + kx < imgWidth) {
                        // Get the filter value and corresponding image pixel
                        float filterValue = BLUR_FILT[(ky + filterHalf) * BLUR_FILTER_WIDTH + (kx + filterHalf)];
                        unsigned char imgPixel = inputImg[(y + ky) * imgWidth + (x + kx)];

                        // Calculate filter sum for normalization
                        filterSum += filterValue;
                        // Accumulate weighted pixel values
                        pixelSum += filterValue * imgPixel;
                    }
                }
            }

            // Normalize and assign new pixel value
            outputImg[y * imgWidth + x] = (unsigned char)(pixelSum / filterSum);
        }
    }
}


int main()
{
    // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    const char filename[] = "C:\\AUT24_GPUComp\\Aut24_GPUCompute\\HW2\\hw2_testimage1.png";
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char* imgData = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
    int imgSize = x_cols * y_rows * sizeof(unsigned char);

    // setup additional host variables, allocate host memory as needed
    unsigned char* h_imgOut = (unsigned char*)malloc(imgSize);

    LARGE_INTEGER frequency;
    LARGE_INTEGER start1, end1, start2, end2, start3, end3;
    if (!QueryPerformanceFrequency(&frequency)) {
        printf("QueryPerformanceFrequency failed!\n");
        return -1;
    }

    if(USE_GPU) {
        // START timer #1
        QueryPerformanceCounter(&start1);

        // allocate device memory
        unsigned char *dev_in = 0;
        unsigned char *dev_out = 0;
        float *dev_filter = 0;
        checkCuda(cudaSetDevice(0));

        checkCuda(cudaMalloc((void**)&dev_in, imgSize));
        checkCuda(cudaMalloc((void**)&dev_out, imgSize));
        checkCuda(cudaMalloc((void**)&dev_filter, BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH * sizeof(float)));

        // copy host data to device
        checkCuda(cudaMemcpy(dev_in, imgData, imgSize, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(dev_filter, BLUR_FILT, BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH * sizeof(float), cudaMemcpyHostToDevice));

        // START timer #2
        QueryPerformanceCounter(&start2);

        // launch kernel --- use appropriate heuristics to determine #threads/block and #blocks/grid to ensure coverage of your 2D data range
        dim3 threadsPerBlock(BLUR_FILTER_WIDTH, BLUR_FILTER_WIDTH);
        dim3 blocksPerGrid((x_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (y_rows + threadsPerBlock.y - 1) / threadsPerBlock.y); 

        int kernel_type = GPU_STATIC;
        if (kernel_type == GPU_GLOBAL) {
            blurKernelGlobal << <blocksPerGrid, threadsPerBlock >> > (
                x_cols, y_rows, dev_filter, dev_in, dev_out);
        }
        if (kernel_type == GPU_STATIC) {
            blurKernelSharedStatic << <blocksPerGrid, threadsPerBlock >> > (
                x_cols, y_rows, dev_filter, dev_in, dev_out);
        }
        if (kernel_type == GPU_DYNAMIC) {
            int shared_memory_size = BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH * sizeof(float);
            blurKernelSharedDynamic << <blocksPerGrid, threadsPerBlock, shared_memory_size >> > (
                x_cols, y_rows, dev_filter, dev_in, dev_out);
        }


        // Check for any errors launching the kernel
        checkCuda(cudaGetLastError());

        // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
        // any errors encountered during the launch.
        checkCuda(cudaDeviceSynchronize());
        
        // STOP timer #2
        QueryPerformanceCounter(&end2);
        double elapsedTime2 = (double)(end2.QuadPart - start2.QuadPart) / frequency.QuadPart;
        printf("Elapsed time: %.6f seconds\n", elapsedTime2);


        // retrieve result data from device back to host
        checkCuda(cudaMemcpy(h_imgOut, dev_out, imgSize, cudaMemcpyDeviceToHost));

        // STOP timer #1
        QueryPerformanceCounter(&end1);
        double elapsedTime1 = (double)(end1.QuadPart - start1.QuadPart) / frequency.QuadPart;
        printf("Elapsed time: %.6f seconds\n", elapsedTime1);

        // cudaDeviceReset( ) must be called in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        checkCuda(cudaDeviceReset());

        // free host and device memory
        cudaFree(dev_filter);
        cudaFree(dev_out);
        cudaFree(dev_in);
        free(imgData);
    } else {
        // EXTRA CREDIT:
        // start timer #3
        QueryPerformanceCounter(&start3);
        // run host sequential blur routine
        cpu_blur(imgData, h_imgOut, x_cols, y_rows);
        // stop timer #3
        QueryPerformanceCounter(&end3);
        double elapsedTime3 = (double)(end3.QuadPart - start3.QuadPart) / frequency.QuadPart;
        printf("Elapsed time: %.6f seconds\n", elapsedTime3);
        free(imgData);
    }

    // save result output image data to file
    const char imgFileOut[] = "C:\\AUT24_GPUComp\\Aut24_GPUCompute\\HW2\\hw2_outimage1.png";
    stbi_write_png(imgFileOut, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);
    free(h_imgOut);
    // retrieve and save timer results (write to console or file)
 
    return 0;
}



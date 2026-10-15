#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define TILE_WIDTH 16

// convolution function
__global__ void convolution_cuda(const uint8_t *image, uint8_t *convoluted, const float *filter, int size, int filterSize) {
    int offset = filterSize / 2;
    
    // threading
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size) {
        float sum = 0.0f;
        for (int f_y = 0; f_y < filterSize; f_y++) {
            for (int f_x = 0; f_x < filterSize; f_x++) {
                int pixel_x = x + f_x - offset;
                int pixel_y = y + f_y - offset;

                if (pixel_x >= 0 && pixel_x < size && pixel_y >= 0 && pixel_y < size) {
                    sum += image[pixel_y * size + pixel_x] * filter[f_y * filterSize + f_x];
                }
            }
        }
        if (sum < 0.0f)   sum = 0.0f;
        if (sum > 255.0f) sum = 255.0f;
        
        convoluted[y * size + x] = (uint8_t)sum;
    }
}

extern "C" void convolution(uint8_t *h_image, uint8_t *h_convoluted, float *h_filter, int size, int filterSize) {
    size_t img_bytes = (size_t)size * size * sizeof(uint8_t);
    size_t filter_bytes = (size_t)(filterSize * filterSize) * sizeof(float);

    uint8_t *d_image, *d_convoluted;
    float *d_filter;

    cudaMalloc((void**)&d_image, img_bytes);
    cudaMalloc((void**)&d_convoluted, img_bytes);
    cudaMalloc((void**)&d_filter, filter_bytes);

    cudaMemcpy(d_image, h_image, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((size + TILE_WIDTH - 1) / TILE_WIDTH, (size + TILE_WIDTH - 1) / TILE_WIDTH);

    convolution_cuda<<<dimGrid, dimBlock>>>(d_image, d_convoluted, d_filter, size, filterSize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_convoluted, d_convoluted, img_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_convoluted);
    cudaFree(d_filter);
}

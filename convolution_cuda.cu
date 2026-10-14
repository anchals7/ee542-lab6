
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

// One GPU thread computes one output pixel.
__global__ void convolve_kernel(
    const uint8_t *input,
    uint8_t *output,
    int M,
    const float *filter,
    int N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= M || y >= M) return;

    int pad = N / 2;
    float sum = 0.0f;

    for (int ky = 0; ky < N; ky++) {
        for (int kx = 0; kx < N; kx++) {
            int iy = y + ky - pad;
            int ix = x + kx - pad;

            if (iy >= 0 && iy < M && ix >= 0 && ix < M) {
                int fy = N - 1 - ky;
                int fx = N - 1 - kx;

                sum += input[iy * M + ix]
                     * filter[fy * N + fx];
            }
        }
    }

    // Match the CPU's absolute-response and clamping behavior.
    sum = fabsf(sum);
    if (sum < 0.0f) sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;

    output[y * M + x] = (uint8_t)(sum + 0.5f);
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 256;
    int N = (argc > 2) ? atoi(argv[2]) : 3;

    if (M <= 0 || N <= 0 || N % 2 == 0 || N > M) {
        fprintf(stderr,
                "Usage: %s [image_size M] [odd_filter_size N]\n",
                argv[0]);
        return 1;
    }

    size_t pixels = (size_t)M * M;
    size_t image_bytes = pixels * sizeof(uint8_t);
    size_t filter_bytes = (size_t)N * N * sizeof(float);

    uint8_t *h_input = (uint8_t *)malloc(image_bytes);
    uint8_t *h_output = (uint8_t *)malloc(image_bytes);
    uint8_t *h_reference = (uint8_t *)malloc(image_bytes);
    float *h_filter = (float *)calloc((size_t)N * N, sizeof(float));

    if (!h_input || !h_output || !h_reference || !h_filter) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    // Deterministic input image.
    srand(0);
    for (size_t i = 0; i < pixels; i++) {
        h_input[i] = (uint8_t)(rand() % 256);
    }

    // Same generalized Laplacian filter as the CPU version.
    int center = N / 2;
    h_filter[center * N + center] = 4.0f;

    if (center > 0) {
        h_filter[(center - 1) * N + center] = -1.0f;
        h_filter[(center + 1) * N + center] = -1.0f;
        h_filter[center * N + center - 1] = -1.0f;
        h_filter[center * N + center + 1] = -1.0f;
    }

    // CPU reference implementation, using identical inputs.
    int pad = N / 2;
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float sum = 0.0f;

            for (int ky = 0; ky < N; ky++) {
                for (int kx = 0; kx < N; kx++) {
                    int iy = y + ky - pad;
                    int ix = x + kx - pad;

                    if (iy >= 0 && iy < M &&
                        ix >= 0 && ix < M) {
                        int fy = N - 1 - ky;
                        int fx = N - 1 - kx;

                        sum += h_input[iy * M + ix]
                             * h_filter[fy * N + fx];
                    }
                }
            }

            sum = fabsf(sum);
            if (sum < 0.0f) sum = 0.0f;
            if (sum > 255.0f) sum = 255.0f;

            h_reference[y * M + x] = (uint8_t)(sum + 0.5f);
        }
    }

    uint8_t *d_input = NULL;
    uint8_t *d_output = NULL;
    float *d_filter = NULL;

    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_filter, filter_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Measure host-to-device transfers.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, filter_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // Warm-up launch, excluded from kernel timing.
    convolve_kernel<<<grid, block>>>(d_input, d_output, M, d_filter, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure kernel execution only.
    CUDA_CHECK(cudaEventRecord(start));
    convolve_kernel<<<grid, block>>>(d_input, d_output, M, d_filter, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // Measure device-to-host transfer.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    // Verify every output pixel.
    size_t mismatches = 0;
    for (size_t i = 0; i < pixels; i++) {
        if (h_output[i] != h_reference[i]) {
            mismatches++;
        }
    }

    printf("CUDA convolution: image=%dx%d filter=%dx%d\n", M, M, N, N);
    printf("H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms\n",
           h2d_ms, kernel_ms, d2h_ms);
    printf("Total measured GPU time: %.3f ms\n",
           h2d_ms + kernel_ms + d2h_ms);
    printf("Verification: %s (%zu mismatches)\n",
           mismatches == 0 ? "PASS" : "FAIL", mismatches);

    // Save output as a binary grayscale PGM image.
    FILE *file = fopen("edge_cuda_output.pgm", "wb");
    if (file) {
        fprintf(file, "P5\n%d %d\n255\n", M, M);
        fwrite(h_output, sizeof(uint8_t), pixels, file);
        fclose(file);
        printf("Output saved to edge_cuda_output.pgm\n");
    } else {
        perror("Could not create CUDA output image");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_filter));

    free(h_input);
    free(h_output);
    free(h_reference);
    free(h_filter);

    return mismatches == 0 ? 0 : 1;
}

#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        goto cleanup;                                              \
    }                                                              \
} while (0)

// One thread computes one output pixel.
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

    sum = fabsf(sum);
    if (sum < 0.0f) sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;

    output[y * M + x] = (uint8_t)(sum + 0.5f);
}

// C-compatible interface callable from Python ctypes.
// Returns 0 on success and -1 on failure.
extern "C" int gpu_convolve(
    const uint8_t *input,
    uint8_t *output,
    int M,
    const float *filter,
    int N
) {
    if (!input || !output || !filter ||
        M <= 0 || N <= 0 || N % 2 == 0 || N > M) {
        return -1;
    }

    // Prevent overflow in the image and filter size calculations.
    if ((size_t)M > SIZE_MAX / (size_t)M ||
        (size_t)M * M > SIZE_MAX / sizeof(uint8_t) ||
        (size_t)N > SIZE_MAX / (size_t)N ||
        (size_t)N * N > SIZE_MAX / sizeof(float)) {
        return -1;
    }

    size_t image_bytes = (size_t)M * M * sizeof(uint8_t);
    size_t filter_bytes = (size_t)N * N * sizeof(float);

    uint8_t *d_input = NULL;
    uint8_t *d_output = NULL;
    float *d_filter = NULL;
    int result = -1;

    cudaError_t err = cudaMalloc(&d_input, image_bytes);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_output, image_bytes);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_filter, filter_bytes);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_input, input, image_bytes,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_filter, filter, filter_bytes,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    {
        dim3 block(16, 16);
        dim3 grid((M + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);

        convolve_kernel<<<grid, block>>>(
            d_input, d_output, M, d_filter, N);

        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto cleanup;
    }

    err = cudaMemcpy(output, d_output, image_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;

    result = 0;

cleanup:
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_filter) cudaFree(d_filter);

    if (result != 0) {
        fprintf(stderr, "gpu_convolve failed: %s\n",
                cudaGetErrorString(
                    err == cudaSuccess
                    ? cudaErrorUnknown : err));
    }

    return result;
}
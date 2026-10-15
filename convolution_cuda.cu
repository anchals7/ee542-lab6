
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <string.h>

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
    int N,
    int absolute_response
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

    // Match selected CPU filer's response behavior
    if (absolute_response) {
         sum = fabsf(sum);
    }
    if (sum < 0.0f) sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;

    output[y * M + x] = (uint8_t)(sum + 0.5f);
}


int create_filter(float *filter, int N, const char *type) {
    int center = N / 2;
    int absolute_response = 0;

    if (strcmp(type, "edge") == 0 ||
        strcmp(type, "sharpen") == 0) {

        filter[center * N + center] =
            (strcmp(type, "edge") == 0) ? 4.0f : 5.0f;

        filter[(center - 1) * N + center] = -1.0f;
        filter[(center + 1) * N + center] = -1.0f;
        filter[center * N + center - 1] = -1.0f;
        filter[center * N + center + 1] = -1.0f;

        absolute_response = (strcmp(type, "edge") == 0);
    }
    else if (strcmp(type, "blur") == 0) {
        double *weights = (double *)malloc(
            (size_t)N * sizeof(double));

        if (!weights) return -1;

        weights[0] = 1.0;
        for (int i = 1; i < N; i++) {
            weights[i] =
                weights[i - 1] * (N - i) / i;
        }

        double total = 0.0;
        for (int i = 0; i < N; i++) {
            total += weights[i];
        }

        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                filter[y * N + x] =
                    (float)((weights[y] * weights[x]) /
                            (total * total));
            }
        }

        free(weights);
    }
    else {
        fprintf(stderr,
                "Unknown filter '%s'. Use edge, blur, or sharpen.\n",
                type);
        return -1;
    }

    return absolute_response;
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 256;
    int N = (argc > 2) ? atoi(argv[2]) : 3;
    const char *filter_type = (argc > 3) ? argv[3] : "edge";

    if (M <= 0 || N < 3 || N % 2 == 0 || N > M) {
        fprintf(stderr,
                "Usage: %s [image_size M] [odd_filter_size N] "
                "[edge|blur|sharpen]\n",
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

    int absolute_response = create_filter(h_filter, N, filter_type);

    if (absolute_response < 0) {
        free(h_input);
        free(h_output);
        free(h_reference);
        free(h_filter);
        return 1;
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

            if (absolute_response) {
                sum = fabsf(sum);
            }
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
    convolve_kernel<<<grid, block>>>(d_input, d_output, M, d_filter, N, absolute_response);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure kernel execution only.
    CUDA_CHECK(cudaEventRecord(start));
    convolve_kernel<<<grid, block>>>(d_input, d_output, M, d_filter, N, absolute_response);
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

    printf("CUDA convolution: image=%dx%d filter=%dx%d type=%s\n", M, M, N, N, filter_type);
    printf("H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms\n",
           h2d_ms, kernel_ms, d2h_ms);
    printf("Total measured GPU time: %.3f ms\n",
           h2d_ms + kernel_ms + d2h_ms);
    printf("Verification: %s (%zu mismatches)\n",
           mismatches == 0 ? "PASS" : "FAIL", mismatches);

    // Save output as a binary grayscale PGM image.
    char output_filename[64];
    snprintf(output_filename, sizeof(output_filename),
         "%s_cuda_output.pgm", filter_type);

    FILE *file = fopen(output_filename, "wb");
    if (file) {
        fprintf(file, "P5\n%d %d\n255\n", M, M);
        fwrite(h_output, sizeof(uint8_t), pixels, file);
        fclose(file);
        printf("Output saved to %s\n", output_filename);
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


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

void convolve_cpu(
    const uint8_t *input,
    uint8_t *output,
    int M,
    const float *filter,
    int N,
    int absolute_response
) {
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

                        sum += input[iy * M + ix]
                             * filter[fy * N + fx];
                    }
                }
            }

            if (absolute_response) {
                sum = fabsf(sum);
            }

            if (sum < 0.0f) sum = 0.0f;
            if (sum > 255.0f) sum = 255.0f;

            output[y * M + x] = (uint8_t)(sum + 0.5f);
        }
    }
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
    size_t filter_elements = (size_t)N * N;

    uint8_t *input = malloc(pixels);
    uint8_t *output = malloc(pixels);
    float *filter = calloc(filter_elements, sizeof(float));

    if (!input || !output || !filter) {
        fprintf(stderr, "Memory allocation failed\n");
        free(input);
        free(output);
        free(filter);
        return 1;
    }

    // Deterministic synthetic grayscale image.
    srand(0);
    for (size_t i = 0; i < pixels; i++) {
        input[i] = (uint8_t)(rand() % 256);
    }

    // Generalized Laplacian edge filter:
    // center = 4, immediate neighbors = -1.
    // Other entries remain zero.
    int center = N / 2;
    filter[center * N + center] = 4.0f;

    if (center > 0) {
        filter[(center - 1) * N + center] = -1.0f;
        filter[(center + 1) * N + center] = -1.0f;
        filter[center * N + center - 1] = -1.0f;
        filter[center * N + center + 1] = -1.0f;
    }

    clock_t start = clock();

    convolve_cpu(input, output, M, filter, N, 1);

    clock_t end = clock();
    double elapsed_ms =
        1000.0 * (double)(end - start) / CLOCKS_PER_SEC;

    printf("CPU convolution: image=%dx%d filter=%dx%d time=%.3f ms\n",
           M, M, N, N, elapsed_ms);

    FILE *file = fopen("edge_output.pgm", "wb");
    if (!file) {
        perror("Could not create output image");
        free(input);
        free(output);
        free(filter);
        return 1;
    }

    fprintf(file, "P5\n%d %d\n255\n", M, M);
    fwrite(output, sizeof(uint8_t), pixels, file);
    fclose(file);

    printf("Output saved to edge_output.pgm\n");

    free(input);
    free(output);
    free(filter);
    return 0;
}
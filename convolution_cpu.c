
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>

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
        double *weights = malloc((size_t)N * sizeof(double));
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

    int absolute_response = create_filter(filter, N, filter_type);

    if (absolute_response < 0) {
        free(input);
        free(output);
        free(filter);
        return 1;
    }
    clock_t start = clock();

    convolve_cpu(input, output, M, filter, N, absolute_response);

    clock_t end = clock();
    double elapsed_ms =
        1000.0 * (double)(end - start) / CLOCKS_PER_SEC;

    printf("CPU convolution: image=%dx%d filter=%dx%d type=%s time=%.3f ms\n", M, M, N, N, filter_type, elapsed_ms);

    char output_filename[64];
    snprintf(output_filename, sizeof(output_filename),"%s_output.pgm", filter_type);

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

    printf("Output saved to %s\n", output_filename);

    free(input);
    free(output);
    free(filter);
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/*
 * Apply an N x N filter to an M x M grayscale image.
 *
 * - Input and output pixels are unsigned 8-bit integers.
 * - Filter coefficients and accumulation use float.
 * - Out-of-bounds pixels are treated as zero.
 * - absolute_response is enabled for edge detection.
 */
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

                        // Flip the filter for mathematical convolution.
                        int fy = N - 1 - ky;
                        int fx = N - 1 - kx;

                        float pixel = input[iy * M + ix];
                        float coeff = filter[fy * N + fx];

                        sum += pixel * coeff;
                    }
                }
            }

            if (absolute_response) {
                sum = fabsf(sum);
            }

            // Convert the result back to the valid pixel range.
            if (sum < 0.0f)   sum = 0.0f;
            if (sum > 255.0f) sum = 255.0f;

            output[y * M + x] = (uint8_t)(sum + 0.5f);
        }
    }
}

int main(void) {
    const int M = 64;

    uint8_t *input = calloc((size_t)M * M, sizeof(uint8_t));
    uint8_t *output = calloc((size_t)M * M, sizeof(uint8_t));

    if (input == NULL || output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(input);
        free(output);
        return 1;
    }

    // Synthetic test image: a bright square on a dark background.
    for (int y = 16; y < 48; y++) {
        for (int x = 16; x < 48; x++) {
            input[y * M + x] = 255;
        }
    }

    const float edge_filter[9] = {
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0
    };

    convolve_cpu(input, output, M, edge_filter, 3, 1);

    // Write the output as a binary grayscale PGM image.
    FILE *file = fopen("edge_output.pgm", "wb");

    if (file == NULL) {
        perror("Could not create output image");
        free(input);
        free(output);
        return 1;
    }

    fprintf(file, "P5\n%d %d\n255\n", M, M);
    fwrite(output, sizeof(uint8_t), (size_t)M * M, file);
    fclose(file);

    printf("CPU convolution completed: %d x %d image, 3 x 3 filter\n", M, M);
    printf("Output saved to edge_output.pgm\n");

    free(input);
    free(output);
    return 0;
}
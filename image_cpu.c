#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// convolution function
void convolution(unsigned char *image, unsigned char *convoluted, float *filter, int size, int filterSize) {
    int offset = filterSize / 2;
    for (int x = 0; x < size; x++){
        for (int y = 0; y < size; y++){
            float sum = 0.0f;
            for (int f_x = 0; f_x < filterSize; f_x++){
                for (int f_y = 0; f_y < filterSize; f_y++){ 
    		    int pixel_x = x + f_x - offset;
    		    int pixel_y = y + f_y - offset;
    		    if (pixel_x >= 0 && pixel_x < size && pixel_y >= 0 && pixel_y < size){
   		        sum += image[pixel_y * size + pixel_x] * filter[f_y * filterSize + f_x];
    		    }
    	        }
            }
        if (sum < 0.0f) {sum = 0.0f;}
        if (sum > 255.0f) {sum = 255.0f;}
        convoluted[y * size + x] = (unsigned char)sum;
        }
    } 
}

int main() {
    //  get images
    const char *image_files[5] = {
        "image1_512.raw", 
        "image2_1024.raw", 
        "image3_2048.raw", 
    };
    int image_sizes[3] = {512, 1024, 2048};

    // set up filters
    int edge_n = 3;
    float edge[9] = {
        -1, -1, -1, 
        -1,  8, -1, 
        -1, -1, -1
    };

    int blur_n = 5;
    float blur[25] = {
        1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f,
        4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f,
        6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f, 6.0f/256.0f,
        4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f,
        1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f
    };

    int sharpen_n = 7;
    float sharpen[49] = {
        -1, -2, -3,  -4, -3, -2, -1,
        -2, -4, -6,  -8, -6, -4, -2,
        -3, -6, -12, -16, -12, -6, -3,
        -4, -8, -16, 217, -16, -8, -4,
        -3, -6, -12, -16, -12, -6, -3,
        -2, -4, -6,  -8, -6, -4, -2,
        -1, -2, -3,  -4, -3, -2, -1
    };

    float *filters[3] = {edge, blur, sharpen};
    int filter_sizes[3] = {edge_n, blur_n, sharpen_n};
    const char *filter_names[3] = {"Edge", "Blur", "Sharpen"};
    size_t max_pixels = (size_t)2048 * 2048;
    size_t max_filter_elements = (size_t)7 * 7;
    uint8_t *input = malloc(max_pixels);
    uint8_t *output = malloc(max_pixels);
    float *filter_buffer = calloc(max_filter_elements, sizeof(float));

    // go through images 
    if (!input || !output || !filter_buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        free(input); free(output); free(filter_buffer);
        return 1;
    }
    for (int img_i = 0; img_i < 3; img_i++) {
        int M = image_sizes[img_i];
        size_t pixels = (size_t)M * M;

        FILE *img_file = fopen(image_files[img_i], "rb");

        size_t read_bytes = fread(input, sizeof(uint8_t), pixels, img_file);
        fclose(img_file);

        if (read_bytes != pixels) {
            fprintf(stderr, "Warning: Size mismatch reading %s\n", image_files[img_i]);
            continue;
        }

        // go through filters
        for (int f_i = 0; f_i < 3; f_i++) {
            int N = filter_sizes[f_i];
            size_t filter_elements = (size_t)(N * N);

            for (size_t k = 0; k < filter_elements; k++) {
                filter_buffer[k] = filters[f_i][k];
            }

            // time convolution function
            clock_t start = clock();
            convolution(input, output, filter_buffer, M, N);
            clock_t end = clock();

            double elapsed_ms = 1000.0 * (double)(end - start) / CLOCKS_PER_SEC;
            printf("Image: %s (M=%d) | Filter: %s (%dx%d) | Time: %.3f ms\n", 
                   image_files[img_i], M, filter_names[f_i], N, N, elapsed_ms);

            // save new images
            char out_filename[128];
            sprintf(out_filename, "out_img%d_%s.pgm", img_i + 1, filter_names[f_i]);
            FILE *out_file = fopen(out_filename, "wb");
            if (out_file) {
                fprintf(out_file, "P5\n%d %d\n255\n", M, M);
                fwrite(output, sizeof(uint8_t), pixels, out_file);
                fclose(out_file);
            }
        }
    }

    // clean up
    free(input);
    free(output);
    free(filter_buffer);

    return 0;
}


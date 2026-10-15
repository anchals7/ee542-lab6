
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

#define CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                cudaGetErrorString(err_), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__global__ void matrixMultiplyTiled(
    const float *A, const float *B, float *C, int N)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numTiles; ++m) {
        int aCol = m * TILE_WIDTH + tx;
        int bRow = m * TILE_WIDTH + ty;

        if (Row < N && aCol < N)
            ds_A[ty][tx] = A[Row * N + aCol];
        else
            ds_A[ty][tx] = 0.0f;

        if (bRow < N && Col < N)
            ds_B[ty][tx] = B[bRow * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int verify = (argc > 2) ? atoi(argv[2]) : 0;

    if (N <= 0) {
        fprintf(stderr, "Matrix size must be positive.\n");
        return 1;
    }

    size_t size = (size_t)N * N * sizeof(float);

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    if (!A || !B || !C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    srand(0);

    for (size_t i = 0; i < (size_t)N * N; ++i) {
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }

    CHECK(cudaFree(0));

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, size));
    CHECK(cudaMalloc(&d_B, size));
    CHECK(cudaMalloc(&d_C, size));

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
              (N + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t e0, e1, e2, e3;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventCreate(&e2));
    CHECK(cudaEventCreate(&e3));

    // Host-to-device transfers
    CHECK(cudaEventRecord(e0));
    CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));

    float ms_h2d, ms_kernel, ms_d2h;

    CHECK(cudaEventElapsedTime(&ms_h2d, e0, e1));

    // Warm-up launch
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Timed kernel launch
    CHECK(cudaEventRecord(e1));
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(e2));

    // Device-to-host transfer
    CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(e3));
    CHECK(cudaEventSynchronize(e3));

    CHECK(cudaEventElapsedTime(&ms_kernel, e1, e2));
    CHECK(cudaEventElapsedTime(&ms_d2h, e2, e3));

    float ms_total = ms_h2d + ms_kernel + ms_d2h;

    printf("Tiled GPU kernel time (N=%d): %.3f ms\n",
           N, ms_kernel);
    printf("H2D copy: %.3f ms | D2H copy: %.3f ms | "
           "kernel + copies: %.3f ms\n",
           ms_h2d, ms_d2h, ms_total);
    printf("RESULT,%d,%.4f,%.4f,%.4f,%.4f\n",
           N, ms_kernel, ms_h2d, ms_d2h, ms_total);

    if (verify) {
        int bad = 0;

        // Same A and B; calculate reference output for sampled elements.
        for (int t = 0; t < 200; ++t) {
            int i = rand() % N;
            int j = rand() % N;
            float ref = 0.0f;

            for (int k = 0; k < N; ++k)
                ref += A[i * N + k] * B[k * N + j];

            if (fabsf(ref - C[i * N + j]) >
                1e-3f * fabsf(ref) + 1e-3f)
                bad++;
        }

        printf("Verification: %s (%d/200 mismatches)\n",
               bad ? "FAILED" : "PASSED", bad);
    }

    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    CHECK(cudaEventDestroy(e2));
    CHECK(cudaEventDestroy(e3));

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(A);
    free(B);
    free(C);

    return 0;
}

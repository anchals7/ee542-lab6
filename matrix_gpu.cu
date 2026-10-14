// matrix_gpu.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                    \
    cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",                         \
                cudaGetErrorString(err_), __FILE__, __LINE__);              \
        exit(1);                                                            \
    }                                                                       \
} while (0)

#define BLOCK 16 //block squared (ie. 16*16=256) is number of threads. tuneable from 1-1024. typically in range of 128-512

__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int verify = (argc > 2) ? atoi(argv[2]) : 0;
    size_t size = (size_t)N * N * sizeof(float);

   //allocate memtory
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);
    srand(0); //passing in 0 to  ensure srand is consistent between runs
   //fill matricies A & B with values 
   for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }

    // Create the CUDA context up front so its one-time cost isn't timed.
    CHECK(cudaFree(0));

    //allocate gpu memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, size));
    CHECK(cudaMalloc(&d_B, size));
    CHECK(cudaMalloc(&d_C, size));

    // allocate the block and grid size
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    cudaEvent_t e0, e1, e2, e3; //use these for timestamping
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventCreate(&e2));
    CHECK(cudaEventCreate(&e3));

    //Host -> Device copies
    CHECK(cudaEventRecord(e0));
    CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms_h2d, ms_kernel, ms_d2h;
    CHECK(cudaEventElapsedTime(&ms_h2d, e0, e1));

    // Warm-up launch (first launch pays module-loading cost) — not timed.
    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    //Kernel runs matrix multiplication
    CHECK(cudaEventRecord(e1));
    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(e2));

    //Device -> Host copy
    CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(e3));
    CHECK(cudaEventSynchronize(e3));

    CHECK(cudaEventElapsedTime(&ms_kernel, e1, e2));
    CHECK(cudaEventElapsedTime(&ms_d2h, e2, e3));

    //calculate and print results
    float ms_total = ms_h2d + ms_kernel + ms_d2h;
    printf("GPU kernel time (N=%d): %.3f ms\n", N, ms_kernel);
    printf("H2D copy: %.3f ms | D2H copy: %.3f ms | kernel + copies: %.3f ms\n",
           ms_h2d, ms_d2h, ms_total);
    //line for automation script: N,kernel_ms,h2d_ms,d2h_ms,total_ms
    printf("RESULT,%d,%.4f,%.4f,%.4f,%.4f\n", N, ms_kernel, ms_h2d, ms_d2h, ms_total);

    if (verify) {
        //Spot-check against cpu if argc[2] == 1
        int bad = 0;
        for (int t = 0; t < 200; t++) {
            int i = rand() % N, j = rand() % N;
            float ref = 0.0f;
            for (int k = 0; k < N; k++) ref += A[i * N + k] * B[k * N + j];
            if (fabsf(ref - C[i * N + j]) > 1e-3f * fabsf(ref) + 1e-3f) bad++;
        }
        printf("Verification: %s (%d/200 mismatches)\n", bad ? "FAILED" : "PASSED", bad);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}

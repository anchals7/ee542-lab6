
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error: %d (%s:%d)\n", \
                (int)status, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

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

    // Deterministic inputs, matching the CUDA implementations.
    srand(0);
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }
    // Pin host buffers to allow more efficient CUDA transfers.
    CHECK_CUDA(cudaHostRegister(A, size, cudaHostRegisterDefault)); 
    CHECK_CUDA(cudaHostRegister(B, size, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(C, size, cudaHostRegisterDefault));

    CHECK_CUDA(cudaFree(0));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t e0, e1, e2, e3;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventCreate(&e2));
    CHECK_CUDA(cudaEventCreate(&e3));

    // Host-to-device transfers.
    CHECK_CUDA(cudaEventRecord(e0));
    CHECK_CUDA(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(e1));
    CHECK_CUDA(cudaEventSynchronize(e1));

    float ms_h2d, ms_kernel, ms_d2h;
    CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, e0, e1));

    // Warm up cuBLAS before timing.
    // Reverse B and A for row-major matrices.
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed cuBLAS matrix multiplication.
    CHECK_CUDA(cudaEventRecord(e1));

    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N
    ));

    CHECK_CUDA(cudaEventRecord(e2));

    // Device-to-host transfer.
    CHECK_CUDA(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(e3));
    CHECK_CUDA(cudaEventSynchronize(e3));

    CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, e1, e2));
    CHECK_CUDA(cudaEventElapsedTime(&ms_d2h, e2, e3));

    float ms_total = ms_h2d + ms_kernel + ms_d2h;

    printf("cuBLAS kernel time (N=%d): %.3f ms\n", N, ms_kernel);
    printf("H2D copy: %.3f ms | D2H copy: %.3f ms | "
           "kernel + copies: %.3f ms\n",
           ms_h2d, ms_d2h, ms_total);
    printf("RESULT,%d,%.4f,%.4f,%.4f,%.4f\n",
           N, ms_kernel, ms_h2d, ms_d2h, ms_total);

    if (verify) {
        int bad = 0;

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

    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaEventDestroy(e2));
    CHECK_CUDA(cudaEventDestroy(e3));

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    CHECK_CUDA(cudaHostUnregister(A));
    CHECK_CUDA(cudaHostUnregister(B));
    CHECK_CUDA(cudaHostUnregister(C));
    
    free(A);
    free(B);
    free(C);

    return 0;
}

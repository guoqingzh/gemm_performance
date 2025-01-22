#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for GEMM
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// Host function to launch the kernel
void gemm(float* A, float* B, float* C, int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}


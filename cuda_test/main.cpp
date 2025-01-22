#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>


// Function to generate a random matrix
std::vector<float> generateMatrix(int rows, int cols) {
    std::vector<float> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(dis(gen));
    }
    return matrix;
}



// Kernel declaration
void gemm(float* A, float* B, float* C, int M, int N, int K);

int main() {
    // Matrix dimensions
    int M = 8192, N = 8192, K = 8192;

    // Host matrices
    std::vector<float> A = generateMatrix(M,K);
    std::vector<float> B = generateMatrix(K,N);
    std::vector<float> C(M * N, 0);

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Create CUDA events for profiling
    cudaEvent_t startMemcpyHtoD, endMemcpyHtoD;
    cudaEvent_t startKernel, endKernel;
    cudaEvent_t startMemcpyDtoH, endMemcpyDtoH;

    cudaEventCreate(&startMemcpyHtoD);
    cudaEventCreate(&endMemcpyHtoD);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&endKernel);
    cudaEventCreate(&startMemcpyDtoH);
    cudaEventCreate(&endMemcpyDtoH);


    // Copy data to device
    cudaEventRecord(startMemcpyHtoD);
    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(endMemcpyHtoD);	
    // Launch the kernel
    int num = 1000;
    while (num--) {
	std::cout << "run:" << num << std::endl;
	cudaEventRecord(startKernel);
    	gemm(d_A, d_B, d_C, M, N, K);
	cudaEventRecord(endKernel);

	cudaEventSynchronize(endKernel);
	float timeKernel;
	cudaEventElapsedTime(&timeKernel, startKernel, endKernel);
	std::cout << "Kernel time" << timeKernel << "ms\n";
    }

    // Copy result back to host
    cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Resultant matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


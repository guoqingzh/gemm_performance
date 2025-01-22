__kernel void gemm_kernel(__global float* A, __global float* B, __global float* C,
                          int M, int N, int K) {
    int row = get_global_id(1); // Y-coordinate
    int col = get_global_id(0); // X-coordinate

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}


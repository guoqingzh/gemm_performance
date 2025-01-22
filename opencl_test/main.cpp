#define CL_HPP_ENABLE_EXCEPTIONS
//#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>

// Utility function to load the OpenCL kernel source
std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file.");
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}


// Function to generate a random matrix
std::vector<float> generateMatrix(int rows, int cols) {
    std::vector<float> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0); // Random values between 0 and 10

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(dis(gen));
    }
    return matrix;
}


int main() {
    // Matrix dimensions
    int M = 8192, N = 8192, K = 8192;

    // Host matrices
    std::vector<float> A = generateMatrix(M,K);
    std::vector<float> B = generateMatrix(K,N);
    std::vector<float> C(M * N, 0);

    // Select platform and device
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    // Create context and queue
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load kernel source
    std::string kernelSource = loadKernelSource("gemm_kernel.cl");
    cl::Program program(context, kernelSource);

    // Build the kernel program
    try {
        program.build();
    } catch (const cl::Error& e) {
        std::cerr << "Build Log:\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    cl::Kernel kernel(program, "gemm_kernel");

    // Allocate device memory
    cl::Buffer d_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M * K * sizeof(float), A.data());
    cl::Buffer d_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K * N * sizeof(float), B.data());
    cl::Buffer d_C(context, CL_MEM_WRITE_ONLY, M * N * sizeof(float));

    // Set kernel arguments
    kernel.setArg(0, d_A);
    kernel.setArg(1, d_B);
    kernel.setArg(2, d_C);
    kernel.setArg(3, M);
    kernel.setArg(4, N);
    kernel.setArg(5, K);

    // Execute the kernel
    int num = 1000;
    while (num--) {
	std::cout << "Run:" << num << std::endl;
	cl::Event event;
	cl::NDRange global(N, M);
   	cl::NDRange local(1, 1); // Optional: Adjust based on device capabilities
    	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &event);
    	queue.finish();
	// Retrieve profiling information
	cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	double executionTimeMs = (endTime - startTime) / 1e6; // Convert to
	std::cout << "Kernel execution time: " << executionTimeMs << " ms\n";
    }

    // Read back results
    queue.enqueueReadBuffer(d_C, CL_TRUE, 0, M * N * sizeof(float), C.data());

    // Print the result
    std::cout << "Resultant matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}


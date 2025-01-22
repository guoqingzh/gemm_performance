cmake_minimum_required(VERSION 3.18)
project(cuda_gemm_project LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Enable CUDA support
enable_language(CUDA)

# Add executable
add_executable(cuda_gemm main.cpp gemm_kernel.cu)

# Set architecture flags for CUDA
set_target_properties(cuda_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "52;60;61;70;75;80;86"
)


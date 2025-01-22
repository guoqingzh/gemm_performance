cmake_minimum_required(VERSION 3.10)
project(opencl_gemm_project LANGUAGES CXX)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Add executable
add_executable(opencl_gemm main.cpp)

# Link OpenCL library
target_include_directories(opencl_gemm PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(opencl_gemm PRIVATE ${OpenCL_LIBRARIES})


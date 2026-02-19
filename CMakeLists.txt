cmake_minimum_required(VERSION 3.16)
project(gdr_rdma LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use modern CUDAToolkit package (adds include + libs cleanly)
find_package(CUDAToolkit REQUIRED)

add_executable(rdma_server rdma_server.cpp)
add_executable(rdma_client rdma_client.cpp)

target_include_directories(rdma_server PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(rdma_client PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(rdma_server PRIVATE ibverbs rdmacm pthread CUDA::cudart CUDA::cuda_driver)
target_link_libraries(rdma_client PRIVATE ibverbs rdmacm pthread CUDA::cudart CUDA::cuda_driver)
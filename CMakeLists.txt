CMAKE_MINIMUM_REQUIRED(VERSION 3.3)

PROJECT(CUDA_FSP)

find_package(CUDA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.0/bin/nvcc)

set(CMAKE_CXX_FLAGS "-fopenmp -O3")

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})

# Where to put .a files (archive for static libraries)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Where to put .so files (dynamic libraries)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Where to put the executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the folders to include
#include_directories(${CMAKE_SOURCE_DIR}/src)
#include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(dummy tests/zero_test.cpp)
target_link_libraries(dummy)

set(TESTS
        get_states_2d
        )

foreach ( PROG ${TESTS})
  cuda_add_executable(${PROG} ${CMAKE_SOURCE_DIR}/tests/${PROG}.cu)
  target_link_libraries(${PROG} -g)
endforeach()
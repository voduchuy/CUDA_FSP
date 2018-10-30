CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

PROJECT(CUDA_FSP LANGUAGES CXX CUDA)

find_package(CUDA)


#set(CMAKE_CUDA_SEPARABLE_COMPILATION true)

set(CMAKE_CXX_FLAGS " -O3 -std=c++11")

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

include_directories(/usr/local/include)

set(CUDA_FSP_LIBS cme_util FSPMat)
foreach (lib ${CUDA_FSP_LIBS})
   add_library(${lib} src/${lib}.h src/${lib}.cu)
   target_compile_features(${lib} PUBLIC cxx_std_11)
   set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach ()
target_link_libraries(FSPMat cme_util)

add_executable(dummy tests/zero_test.cpp)
target_link_libraries(dummy)

set(TESTS
        get_states_3d
        gen_mat_set_toggle
        )

foreach (PROG ${TESTS})
    cuda_add_executable(${PROG} ${CMAKE_SOURCE_DIR}/tests/${PROG}.cu)
    target_link_libraries(${PROG})
    target_compile_features(${PROG} PUBLIC cxx_std_11)
    set_target_properties(${PROG} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach ()
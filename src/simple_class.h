//
// Created by Huy Vo on 10/31/18.
//

#ifndef CUDA_FSP_SIMPLE_CLASS_H
#define CUDA_FSP_SIMPLE_CLASS_H
#include <cuda_runtime.h>
#include <iostream>

#define CUDACHKERR() { \
cudaError_t ierr = cudaGetLastError();\
if (ierr != cudaSuccess){ \
    printf("%s in %s at line %d\n", cudaGetErrorString(ierr), __FILE__, __LINE__);\
    exit(EXIT_FAILURE); \
}\
}\

namespace abc{

    __global__
    void print_kernel(int x);

    class simple_class {
        int data = 0;
    public:
        simple_class(int x);
    };
}




#endif //CUDA_FSP_SIMPLE_CLASS_H

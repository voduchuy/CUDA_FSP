//
// Created by Huy Vo on 10/31/18.
//
#include "simple_class.h"

namespace abc{
    __global__
    void print_kernel(int x){
        extern __shared__ int wsp[];
        printf("x= %d \n", x);
    }

    simple_class::simple_class(int x) {

        data = x;

        for (int i{0}; i < 10; ++i){
            print_kernel<<<1, 16, 128>>>(i); CUDACHKERR();
            cudaDeviceSynchronize();
        }

        cudaDeviceSynchronize();
    }
}
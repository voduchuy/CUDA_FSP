//
// Created by huy on 10/24/18.
//
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <assert.h>
#include <time.h>
#include "driver_types.h"
#include "cme_util.h"

using namespace cuFSP;

int main() {
    int d = 3;
    int *d_states;
    int *states_cpu;
    int *n_bounds;
    int *d_n_bounds;

    n_bounds = new int[d];
    n_bounds[0] = 1023;
    n_bounds[1] = 1023;
    n_bounds[2] = 3;

    cudaMalloc((void **) &d_n_bounds, d * sizeof(int));
    cudaMemcpy(d_n_bounds, n_bounds, d * sizeof(int), cudaMemcpyHostToDevice);

    int n_states = cuFSP::rect_fsp_num_states(d, n_bounds);

    cudaMallocManaged((void **) &d_states, d * n_states * sizeof(int)); CUDACHKERR();
    fsp_get_states << < (size_t) std::ceil(n_states / (32.0)), 32, 1024 >> > (d_states, d, n_states, d_n_bounds);
    cudaDeviceSynchronize();
    CUDACHKERR();

    states_cpu = new int[n_states*d*sizeof(int)];
    cudaMemcpy(states_cpu, d_states, n_states*d*sizeof(int), cudaMemcpyDeviceToHost);
    CUDACHKERR();

    bool success = true;

    int i = 0;
    for (int x2{0}; x2 <= n_bounds[2]; ++x2){
        for (int x1{0}; x1 <= n_bounds[1]; ++x1){
            for (int x0{0}; x0 <= n_bounds[0]; ++x0){
                success = success && (x0 == states_cpu[i*d])
                                  && (x1 == states_cpu[i*d + 1])
                                  && (x2 == states_cpu[i*d + 2]);
                i++;
            }
        }
    }

    assert(success);
    std::cout << "Test FSP state generation successful.\n";

    cudaFree(d_n_bounds);
    cudaFree(d_states);
    delete[] states_cpu;
    delete[] n_bounds;
    return 0;
}



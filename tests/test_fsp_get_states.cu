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

void get_states_cpu(int *states, size_t dim, int n_states, size_t *n_bounds) {
    for (size_t indx = 0; indx < n_states; ++indx) {
        indx2state(indx, &states[indx * dim], dim, n_bounds);
    }
}

int main() {
    clock_t t1, t2;

    size_t d = 3;

    size_t *n_bounds = new size_t[d];
    n_bounds[0] = 1023;
    n_bounds[1] = 1023;
    n_bounds[2] = 3;

    size_t *d_n_bounds;
    cudaMalloc((void **) &d_n_bounds, d * sizeof(size_t));
    cudaMemcpy(d_n_bounds, n_bounds, d * sizeof(size_t), cudaMemcpyHostToDevice);

    size_t n_states = 1;
    for (size_t i{0}; i < d; ++i) {
        n_states *= (n_bounds[i] + 1);
    }
//    std::cout << "Total number of states:" << n_states << "\n";

    int *d_states, *states_cpu;

    states_cpu = new int[d*n_states];
    t1 = clock();
    get_states_cpu(states_cpu, d, n_states, n_bounds);
    t2 = clock();
    std::cout << "Generate states with CPU take " << (double) (t2 - t1) / CLOCKS_PER_SEC * 1000.0 << " ms.\n";

    cudaMallocManaged((void **) &d_states, d * n_states * sizeof(int)); CUDACHKERR();
    t1 = clock();
    fsp_get_states << < (size_t) std::ceil(n_states / (32.0)), 32, 1024 >> > (d_states, d, n_states, d_n_bounds);
    cudaDeviceSynchronize();
    CUDACHKERR();
    t2 = clock();

    std::cout << "Generate states with GPU take " << (double) (t2 - t1) / CLOCKS_PER_SEC * 1000.0 << " ms.\n";

    cudaDeviceSynchronize();

    bool success = true;
    for (size_t i{0}; i < n_states; ++i) {
        for (size_t k{0}; k < d; ++k) {
            if (d_states[i * d + k] != states_cpu[i * d + k]) {
                success = false;
                break;
            }
        }
    }

    assert(success == true);

    cudaFree(d_n_bounds);
    cudaFree(d_states);
    delete[] states_cpu;
    delete[] n_bounds;
    return 0;
}



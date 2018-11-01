//
// Created by Huy Vo on 10/30/18.
//
#include <armadillo>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "FSPMat.h"
#include "cme_util.h"
#include "FSPMat.h"
#include "cusparse.h"
#include "../src/cme_util.h"
#include "../src/FSPMat.h"

__device__ __host__
double toggle_propensity(int *x, int reaction) {
    double prop_val;
    switch (reaction) {
        case 0:
            prop_val = 1.0 / (1.0 + std::pow(1.0 * x[1], 2.0));
            break;
        case 1:
            prop_val = 1.0 * x[0];
            break;
        case 2:
            prop_val = 1.0 / (1.0 + std::pow(1.0 * x[0], 2.0));
            break;
        case 3:
            prop_val = 1.0 * x[1];
            break;
    }
    return prop_val;
}

__device__ cuFSP::PropFun prop_pointer = &toggle_propensity;

Col<double> t_func(double t){
    return arma::Col<double>({1.0, 1.0, 1.0, 1.0});
}

int main()
{
//    cusparseHandle_t cusparse_handle;
//    cudaStream_t stream;
//
//    // Initialize cuSparse handle and bind to stream
//    cusparseCreate(&cusparse_handle); CUDACHKERR();
//    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); CUDACHKERR();
//    cusparseSetStream(cusparse_handle, stream); CUDACHKERR();

    cudaDeviceSynchronize();

    size_t n_species = 2;
    size_t n_reactions = 4;

    int stoich_vals[] = {1, -1, 1, -1};
    int stoich_colidxs[] = {0, 0, 1, 1};
    int stoich_rowptrs[] = {0, 1, 2, 3, 4};

    cuFSP::cuda_csr_mat_int stoich;
    stoich.vals = &stoich_vals[0];
    stoich.col_idxs = &stoich_colidxs[0];
    stoich.row_ptrs = &stoich_rowptrs[0];
    stoich.n_rows = 4;
    stoich.n_cols = 2;

    size_t *n_bounds;
    int *states;

    cudaMallocManaged(&n_bounds, n_species*sizeof(size_t));

    n_bounds[0] = (1 << 10) - 1;
    n_bounds[1] = (1 << 10) - 1;

    std::cout << n_bounds[0] << " " << n_bounds[1] << "\n";

    size_t n_states = 1;
    for (size_t i{0}; i < n_species; ++i) {
        n_states *= (n_bounds[i] + 1);
    }
    std::cout << "Total number of states:" << n_states << "\n";

    cudaMallocManaged(&states, n_states * n_species * sizeof(int)); CUDACHKERR();

    cuFSP::PropFun host_prop_ptr;
    cudaMemcpyFromSymbol(&host_prop_ptr, prop_pointer, sizeof(cuFSP::PropFun)); CUDACHKERR();

    cuFSP::FSPMat A
//    (cusparse_handle, stream,
                    (states, n_states, n_reactions, n_species, n_bounds,
            stoich, t_func, host_prop_ptr);

    cudaDeviceSynchronize();
    std::cout << "Matrix generation successful.\n";

//    cusparseDestroy(cusparse_handle); CUDACHKERR();
//    cudaStreamDestroy(stream); CUDACHKERR();
    cudaFree(states); CUDACHKERR();
    cudaFree(n_bounds); CUDACHKERR();
//    cudaDeviceReset();
    return 0;
}
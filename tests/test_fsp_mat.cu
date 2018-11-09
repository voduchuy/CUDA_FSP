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
#include "thrust/transform.h"
#include "thrust/execution_policy.h"
#include "thrust/device_vector.h"

/* Parameters for the propensity functions */
const double ayx{2.6e-3}, axy{6.1e-3}, nyx{3e0}, nxy{2.1e0},
        kx0{2.2e-3}, kx{1.7e-2}, dx{3.8e-4}, ky0{6.8e-5}, ky{1.6e-2}, dy{3.8e-4};

__device__ __host__
double toggle_propensity(int *x, int reaction) {
    double prop_val;
    switch (reaction) {
        case 0:
            prop_val = 1.0 / (1.0 + ayx*std::pow(1.0 * x[1], nyx));
            break;
        case 1:
            prop_val = 1.0 * x[0];
            break;
        case 2:
            prop_val = 1.0 / (1.0 + axy*std::pow(1.0 * x[0], nxy));
            break;
        case 3:
            prop_val = 1.0 * x[1];
            break;
    }
    return prop_val;
}

__device__ cuFSP::PropFun prop_pointer = &toggle_propensity;

__device__ __host__
void t_func(double t, double* out){
//    return {(1.0 + std::cos(t))*kx0, kx, dx, (1.0 + std::sin(t))*ky0, ky, dy};
    out[0] = kx0;
    out[1] = kx;
    out[2] = dx;
    out[3] = ky0;
    out[4] = ky;
    out[5] = dy;
}

int main()
{
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

    cudaMalloc(&states, n_states * n_species * sizeof(int)); CUDACHKERR();

    cuFSP::PropFun host_prop_ptr;
    cudaMemcpyFromSymbol(&host_prop_ptr, prop_pointer, sizeof(cuFSP::PropFun)); CUDACHKERR();

    cuFSP::FSPMat A
    (states, n_states, n_reactions, n_species, n_bounds,
            stoich, &t_func, host_prop_ptr);

    cudaDeviceSynchronize();
    std::cout << "Matrix generation successful.\n";

    thrust::device_vector<double> v(n_states);
    thrust::device_vector<double> w(n_states);
    thrust::fill(v.begin(), v.end(), 0.0); CUDACHKERR();
    cudaDeviceSynchronize(); CUDACHKERR();

    A.action(1.0, v, w);

    double sum = thrust::reduce(w.begin(), w.end());

    cudaFree(states); CUDACHKERR();
    cudaFree(n_bounds); CUDACHKERR();

    assert( std::abs(sum) <= 1.0e-14);

    return 0;
}
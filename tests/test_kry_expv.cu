//
// Created by Huy Vo on 10/30/18.
//
#include <armadillo>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cuda_runtime.h>
#include "FSPMat.h"
#include "cme_util.h"
#include "FSPMat.h"
#include "cusparse.h"
#include "thrust/transform.h"
#include "thrust/execution_policy.h"
#include "thrust/device_vector.h"
#include "cme_util.h"
#include "FSPMat.h"
#include "KryExpvFSP.h"

/* Parameters for the propensity functions */
const double ayx{2.6e-3}, axy{6.1e-3}, nyx{3e0}, nxy{2.1e0},
        kx0{2.2e-3}, kx{1.7e-2}, dx{3.8e-4}, ky0{6.8e-5}, ky{1.6e-2}, dy{3.8e-4};

__device__ __host__
double toggle_propensity(int *x, int reaction) {
    double prop_val;
    switch (reaction) {
        case 0:
            prop_val = 1.0;
            break;
        case 1:
            prop_val = 1.0 / (1.0 + ayx*std::pow(1.0 * x[1], nyx));
            break;
        case 2:
            prop_val = 1.0 * x[0];
            break;
        case 3:
            prop_val = 1.0;
            break;
        case 4:
            prop_val = 1.0 / (1.0 + axy*std::pow(1.0 * x[0], nxy));
            break;
        case 5:
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
    int n_species = 2;
    int n_reactions = 6;

    int stoich_vals[] = {1, 1, -1, 1,1, -1};
    int stoich_colidxs[] = {0, 0, 0, 1, 1, 1};
    int stoich_rowptrs[] = {0, 1, 2, 3, 4, 5,6};

    cuFSP::CSRMatInt stoich;
    stoich.vals = &stoich_vals[0];
    stoich.col_idxs = &stoich_colidxs[0];
    stoich.row_ptrs = &stoich_rowptrs[0];
    stoich.n_rows = 6;
    stoich.n_cols = 2;
    stoich.nnz = 6;

    int n_bounds[] = {1<<8, 1<<8};

    std::cout << n_bounds[0] << " " << n_bounds[1] << "\n";
    int n_states = cuFSP::rect_fsp_num_states(n_species, n_bounds);
    std::cout << "Total number of states:" << n_states << "\n";

    cuFSP::PropFun host_prop_ptr;
    cudaMemcpyFromSymbol(&host_prop_ptr, prop_pointer, sizeof(cuFSP::PropFun)); CUDACHKERR();
    cuFSP::FSPMat A
    (n_reactions, n_species, n_bounds,
            stoich, &t_func, host_prop_ptr, cuFSP::HYB); CUDACHKERR();
    cudaDeviceSynchronize();

    thrust::device_vector<double> v(n_states);
    thrust::fill(v.begin(), v.end(), 0.0); CUDACHKERR();
    v[0] = 1.0;
    cudaDeviceSynchronize(); CUDACHKERR();

    double t_final = 8*3600;
    double tol = 1.0e-8;
    int m = 30;
    std::function<void (double*, double*)> matvec = [&] (double*x, double* y) {
        A.action(1.0, x, y);
        return;
    };

    cuFSP::KryExpvFSP expv(t_final, matvec, v, m, tol, true);
    expv.solve();
    cudaDeviceSynchronize();
    double vsum = thrust::reduce(v.begin(), v.end());
    assert(std::abs(1.0 - vsum) <= 1.0e-10);

    return 0;
}
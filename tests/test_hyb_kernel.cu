//
// Created by Huy Vo on 11/16/18.
//
#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include "cme_util.h"


using namespace cuFSP;


__device__ __host__
double bd_propensity(int *x, int reaction) {
    double prop_val;
    switch (reaction) {
        case 0:
            prop_val = 1.0;
            break;
        case 1:
            prop_val = 1.0*x[0];
            break;
        case 2:
            prop_val = 1.0;
            break;
        case 3:
            prop_val = 1.0 * x[1];
            break;
    }
    return prop_val;
}

__device__ cuFSP::PropFun prop_pointer = &bd_propensity;

__device__ __host__
void t_func(double t, double* out){
//    return {(1.0 + std::cos(t))*kx0, kx, dx, (1.0 + std::sin(t))*ky0, ky, dy};
    out[0] = 1.0;
    out[1] = 1.0;
    out[2] = 1.0;
    out[3] = 1.0;
}

__global__
void fspmat_hyb_fill_data(int n_species, int n_reactions, int n_states, int *fsp_bounds, int *states, cuFSP::CSRMatInt stoich,
        cuFSP::PropFun propensity,
        double *diag_vals, double *offdiag_vals, int *offdiag_colindxs){
    extern __shared__ int wsp[];
    int tix = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + tix;

    int *fsp_bounds_copy = wsp;
    int stoich_nnz = stoich.row_ptrs[n_reactions];
    int *stoich_vals = &wsp[n_species];
    int *stoich_colidxs = &stoich_vals[stoich_nnz];
    int *stoich_rowptrs = &stoich_colidxs[stoich_nnz];

    if (tix < n_species) {
        fsp_bounds_copy[tix] = fsp_bounds[tix];
    }
    __syncthreads();

    if (tix < n_reactions+1){
        stoich_rowptrs[tix] = stoich.row_ptrs[tix];
    }

    if (tix < stoich_nnz){
        stoich_vals[tix] = stoich.vals[tix];
        stoich_colidxs[tix] = stoich.col_idxs[tix];
    }
    __syncthreads();

    int *state;

    if (tid < n_states) {
        for (int reaction{0}; reaction < n_reactions; ++reaction){
            state = &states[tid * n_species];
            diag_vals[n_states*reaction + tid] = -1.0*propensity(state, reaction);

            // Fill the off-diagonal entries
            reachable_state(state, state, reaction, -1,
                            n_species, stoich_vals, stoich_colidxs, stoich_rowptrs);

            bool reachable = true;
            for (size_t k{0}; k < n_species; ++k) {
                reachable = reachable && ((state[k] >= 0) && (state[k] <= fsp_bounds_copy[k]));
            }

            if (reachable) {
                offdiag_colindxs[n_states*reaction + tid] = state2indx(state, n_species, fsp_bounds_copy);
                offdiag_vals[n_states*reaction + tid] = propensity(state, reaction);
            } else {
                offdiag_colindxs[n_states*reaction + tid] = -1;
                offdiag_vals[n_states*reaction + tid] = 0.0;
            }

            reachable_state(state, state, reaction, 1,
                            n_species, stoich_vals, stoich_colidxs, stoich_rowptrs);
        }
    }
}

__global__
void fspmat_hyb_mv(int n_states, int n_reactions, double *diag_vals, double *offdiag_vals, int *offdiag_colidxs,
                   double *coef, double *x, double *y){
    double y_val = 0.0;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_states){
        for (int reaction = 0; reaction < n_reactions; ++reaction){
            y_val += coef[reaction]*diag_vals[n_states*reaction + tid]*x[tid];
            __syncthreads();
            y_val += coef[reaction]*offdiag_vals[n_states*reaction + tid]*x[offdiag_colidxs[n_states*reaction + tid]];
            __syncthreads();
        }
        y[tid] = y_val;
    }
}

int main() {
    int n_species = 2;
    int n_reactions = 4;

    int *n_bounds;

    double *diag_vals;
    double *offdiag_vals;
    int *offdiag_colindxs;
    double *x, *y;

    int stoich_vals[] = {1, -1, 1, -1};
    int stoich_colidxs[] = {0, 0, 1, 1};
    int stoich_rowptrs[] = {0, 1, 2, 3, 4};

    int *d_stoich_vals, *d_stoich_colidxs, *d_stoich_rowptrs;
    double *tcoef;
    cudaMallocManaged(&tcoef, 4*sizeof(double)); CUDACHKERR();
    cudaMalloc(&d_stoich_vals, 4*sizeof(int)); CUDACHKERR();
    cudaMalloc(&d_stoich_colidxs, 4*sizeof(int)); CUDACHKERR();
    cudaMalloc(&d_stoich_rowptrs, 5*sizeof(int)); CUDACHKERR();

    cudaMemcpy(d_stoich_vals, stoich_vals, 4*sizeof(int), cudaMemcpyHostToDevice); CUDACHKERR();
    cudaMemcpy(d_stoich_colidxs, stoich_colidxs, 4*sizeof(int), cudaMemcpyHostToDevice); CUDACHKERR();
    cudaMemcpy(d_stoich_rowptrs, stoich_rowptrs, 5*sizeof(int), cudaMemcpyHostToDevice); CUDACHKERR();

    cuFSP::CSRMatInt stoich;
    stoich.vals = &d_stoich_vals[0];
    stoich.col_idxs = &d_stoich_colidxs[0];
    stoich.row_ptrs = &d_stoich_rowptrs[0];
    stoich.n_rows = 4;
    stoich.n_cols = 2;

    cudaMallocManaged(&n_bounds, n_species * sizeof(int));

    n_bounds[0] = (1 << 12) - 1;
    n_bounds[1] = (1 << 12) - 1;

    std::cout << n_bounds[0] << " " << n_bounds[1] << "\n";

    int n_states = cuFSP::rect_fsp_num_states(n_species, n_bounds);
    std::cout << "Total number of states:" << n_states << "\n";

    int *states;
    cudaMallocManaged(&states, n_states * n_species * sizeof(int));
    CUDACHKERR();
    cudaMallocManaged(&diag_vals, n_states * n_reactions * sizeof(double));
    CUDACHKERR();
    cudaMallocManaged(&offdiag_vals, n_states * n_reactions * sizeof(double));
    CUDACHKERR();
    cudaMallocManaged(&offdiag_colindxs, n_states * n_reactions * sizeof(int));
    CUDACHKERR();
    cudaMallocManaged(&x, n_states*sizeof(double)); CUDACHKERR();
    cudaMallocManaged(&y, n_states*sizeof(double)); CUDACHKERR();

    thrust::device_ptr<double> x_ptr(x);
    thrust::fill(x_ptr, x_ptr + n_states, 1.0);
    cudaDeviceSynchronize();
    CUDACHKERR();

    int numBlocks = std::ceil(n_states/1024.0);
    fsp_get_states<<<numBlocks, 1024, n_species*sizeof(int)>>>(states, n_species, n_states, n_bounds);
    cudaDeviceSynchronize();
    CUDACHKERR();

    cuFSP::PropFun host_prop_ptr;
    cudaMemcpyFromSymbol(&host_prop_ptr, prop_pointer, sizeof(cuFSP::PropFun)); CUDACHKERR();
    int shared_mem_size = n_species*sizeof(int) + (stoich.nnz*2 + stoich.n_rows+1)*sizeof(int);
    fspmat_hyb_fill_data<<<numBlocks, 1024, shared_mem_size>>>(n_species, n_reactions, n_states, n_bounds, states, stoich, host_prop_ptr, diag_vals, offdiag_vals, offdiag_colindxs);
    cudaDeviceSynchronize();
    CUDACHKERR();

    t_func(0.0, tcoef);
    fspmat_hyb_mv<<<numBlocks, 1024>>>(n_states, n_reactions, diag_vals, offdiag_vals, offdiag_colindxs, tcoef, x, y);
    cudaDeviceSynchronize();
    CUDACHKERR();
//
//    for (int i{0}; i < n_states; ++i){
//        for (int j{0}; j < n_reactions; ++j){
//            printf("%.2e ", diag_vals[n_states*j + i]);
//        }
//        for (int j{0}; j < n_reactions; ++j){
//            printf("%.2e %d ", offdiag_vals[n_states*j + i], offdiag_colindxs[n_states*j + i]);
//        }
//        printf("\n");
//    }
//
//    for (int i{0}; i < n_states; ++i){
//        printf("y[%d] = %.2e \n", i, y[i]);
//    }

    cudaFree(tcoef); CUDACHKERR();
    cudaFree(d_stoich_colidxs); CUDACHKERR();
    cudaFree(d_stoich_vals); CUDACHKERR();
    cudaFree(d_stoich_rowptrs); CUDACHKERR();
    cudaFree(diag_vals); CUDACHKERR();
    cudaFree(offdiag_vals); CUDACHKERR();
    cudaFree(offdiag_colindxs); CUDACHKERR();
    cudaFree(n_bounds); CUDACHKERR();
    cudaFree(states); CUDACHKERR();
    return 0;
}

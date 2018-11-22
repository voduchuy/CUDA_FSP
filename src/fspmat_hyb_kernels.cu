//
// Created by Huy Vo on 11/21/18.
//
#include "fspmat_hyb_kernels.h"

namespace cuFSP{
    __global__
    void fspmat_hyb_fill_data(int n_species, int n_reactions, int n_states, int *fsp_bounds, int *states, cuFSP::CSRMatInt stoich,
                              cuFSP::PropFun propensity,
                              double *diag_vals, double *offdiag_vals, int *offdiag_colindxs){
        extern __shared__ int wsp[];
        int tix = threadIdx.x;
        int tid = blockDim.x * blockIdx.x + tix;

        int stoich_nnz = stoich.nnz;
        int *fsp_bounds_copy = wsp;
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
        __syncthreads();
        if (tix < stoich_nnz){
            stoich_vals[tix] = stoich.vals[tix];
            stoich_colidxs[tix] = stoich.col_idxs[tix];
        }
        __syncthreads();

        int *state;

        if (tid < n_states) {
            state = &states[tid * n_species];
            for (int reaction{0}; reaction < n_reactions; ++reaction){
                diag_vals[n_states*reaction + tid] = -1.0*propensity(state, reaction);
                // Fill the off-diagonal entries
                reachable_state(state, state, reaction, -1,
                                n_species, stoich_vals, stoich_colidxs, stoich_rowptrs);

                bool reachable = true;
                for (int k{0}; k < n_species; ++k) {
                    reachable = reachable && ((state[k] >= 0) && (state[k] <= fsp_bounds_copy[k]));
                }

                if (reachable) {
                    offdiag_colindxs[n_states*reaction + tid] = state2indx(state, n_species, fsp_bounds_copy);
                    offdiag_vals[n_states*reaction + tid] = propensity(state, reaction);
                } else {
                    offdiag_colindxs[n_states*reaction + tid] = 0;
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
            y[tid] += y_val;
        }
    }

    void HYBMatSet::destroy() {
        if (diag_vals) cudaFree(diag_vals); CUDACHKERR();
        if (offdiag_vals) cudaFree(offdiag_vals); CUDACHKERR();
        if (offdiag_colidxs) cudaFree(offdiag_colidxs); CUDACHKERR();
        if (d_coefs) cudaFree(d_coefs); CUDACHKERR();
    }

    void HYBMatSet::action(double *x, double *y, double *coefs) {
        cudaMemcpy(d_coefs, coefs, num_matrices*sizeof(double), cudaMemcpyHostToDevice);
        CUDACHKERR();

        // Get the max number of threads that can fit to a block
        int max_block_size, num_blocks;
        int device_id;
        cudaGetDevice(&device_id);
        CUDACHKERR();
        cudaDeviceGetAttribute(&max_block_size, cudaDevAttrMaxThreadsPerBlock, device_id);
        CUDACHKERR();

        num_blocks = (int) std::ceil(n_rows/(max_block_size*1.0));
        fspmat_hyb_mv<<<num_blocks, max_block_size>>>(n_rows, num_matrices, diag_vals,
                offdiag_vals, offdiag_colidxs, d_coefs, x, y);
        cudaDeviceSynchronize();
        CUDACHKERR();
    }

    void generate_fsp_mats_hyb(int *states, int n_states, int n_reactions, int n_species, int *fsp_bounds,
                               CSRMatInt stoich, PropFun prop_func, HYBMatSet *hyb){
        int *d_stoich_vals, *d_stoich_colidxs, *d_stoich_rowptrs;
        cudaMalloc(&d_stoich_vals, stoich.nnz * sizeof(int));
        CUDACHKERR();
        cudaMalloc(&d_stoich_colidxs, stoich.nnz * sizeof(int));
        CUDACHKERR();
        cudaMalloc(&d_stoich_rowptrs, (stoich.n_rows + 1) * sizeof(int));
        CUDACHKERR();

        cudaMemcpy(d_stoich_vals, stoich.vals, stoich.nnz * sizeof(int), cudaMemcpyHostToDevice);
        CUDACHKERR();
        cudaMemcpy(d_stoich_colidxs, stoich.col_idxs, stoich.nnz * sizeof(int),
                   cudaMemcpyHostToDevice);
        CUDACHKERR();
        cudaMemcpy(d_stoich_rowptrs, stoich.row_ptrs, (stoich.n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        CUDACHKERR();

        CSRMatInt d_stoich;
        d_stoich.n_rows = stoich.n_rows;
        d_stoich.nnz = stoich.nnz;
        d_stoich.n_cols = stoich.n_cols;
        d_stoich.vals = d_stoich_vals;
        d_stoich.col_idxs = d_stoich_colidxs;
        d_stoich.row_ptrs = d_stoich_rowptrs;

        cudaMalloc(&(hyb->diag_vals), n_states * n_reactions * sizeof(double));
        CUDACHKERR();
        cudaMalloc(&(hyb->offdiag_vals), n_states * n_reactions * sizeof(double));
        CUDACHKERR();
        cudaMalloc(&(hyb->offdiag_colidxs), n_states * n_reactions * sizeof(int));
        CUDACHKERR();

        // Get the max number of threads that can fit to a block
        int max_block_size, num_blocks;
        int device_id;

        cudaGetDevice(&device_id);
        CUDACHKERR();
        cudaDeviceGetAttribute(&max_block_size, cudaDevAttrMaxThreadsPerBlock, device_id);
        CUDACHKERR();

        // Generate data for hyb
        hyb->n_rows = n_states;
        hyb->num_matrices = n_reactions;
        cudaMalloc(&(hyb->d_coefs), hyb->num_matrices*sizeof(double)); CUDACHKERR();

        num_blocks = (int) std::ceil(n_states/(max_block_size*1.0));

        fsp_get_states<<<num_blocks, max_block_size, n_species*sizeof(int)>>>(states, n_species, n_states, fsp_bounds);
        cudaDeviceSynchronize();
        CUDACHKERR();
        int shared_mem_size = n_species*sizeof(int) + (stoich.nnz*2 + stoich.n_rows+1)*sizeof(int);
        fspmat_hyb_fill_data<<<num_blocks, max_block_size, shared_mem_size>>>(n_species, n_reactions, n_states, fsp_bounds, states, d_stoich,
                prop_func, hyb->diag_vals, hyb->offdiag_vals, hyb->offdiag_colidxs);
        cudaDeviceSynchronize();
        CUDACHKERR();

        cudaFree(d_stoich_colidxs);
        CUDACHKERR();
        cudaFree(d_stoich_rowptrs);
        CUDACHKERR();
        cudaFree(d_stoich_vals);
        CUDACHKERR();
    }
}
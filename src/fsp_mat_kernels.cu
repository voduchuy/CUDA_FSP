//
// Created by huy on 11/3/18.
//
#include "fsp_mat_kernels.h"
namespace cuFSP{

    __global__
    void fspmat_component_get_nnz_per_row(int *nnz_per_row, int *off_indx, int *states, int reaction, size_t n_rows,
                                          size_t n_species, size_t *fsp_bounds,
                                          int *stoich_vals, int *stoich_colidxs, int *stoich_rowptrs) {
        extern __shared__ size_t wsp[];

        size_t tix = threadIdx.x;
        size_t tid = blockDim.x * blockIdx.x + tix;

        size_t *fsp_bounds_copy = &wsp[0];

        if (tix < n_species) {
            fsp_bounds_copy[tix] = fsp_bounds[tix];
        }

        __syncthreads();

        int *state;

        if (tid < n_rows) {

            state = &states[tid * n_species];

            indx2state(tid, &state[0], n_species, fsp_bounds_copy);

            reachable_state(state, state, reaction, -1,
                            n_species, stoich_vals, stoich_colidxs, stoich_rowptrs);

            bool reachable = true;
            for (size_t k{0}; k < n_species; ++k) {
                reachable = reachable && ((state[k] >= 0) || (state[k] <= fsp_bounds_copy[k]));
            }

            nnz_per_row[tid] = 1;
            if (reachable) {
                off_indx[tid] = state2indx(state, n_species, fsp_bounds_copy);
                nnz_per_row[tid] += 1;
            } else {
                off_indx[tid] = -1;
            }

            reachable_state(state, state, reaction, 1,
                            n_species, stoich_vals, stoich_colidxs, stoich_rowptrs);
        }
    }

    __global__

    void
    fspmat_component_fill_data_csr(double *values, int *col_indices, int *row_ptrs, size_t n_rows, int reaction,
                                   int *off_diag_indices, int *states, size_t dim, PropFun propensity) {

        size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        int off_diag_indx, rowptr, i_diag, i_offdiag;
        int *state;

        if (tid < n_rows) {
            off_diag_indx = off_diag_indices[tid];
            rowptr = row_ptrs[tid];

            if (off_diag_indx >= 0) {

                if (off_diag_indx > tid) {
                    i_diag = rowptr;
                    i_offdiag = rowptr + 1;
                } else {
                    i_diag = rowptr + 1;
                    i_offdiag = rowptr;
                }

                state = states + dim * tid;
                values[i_diag] = propensity(state, reaction);
                values[i_diag] *= -1.0;

                state = states + dim * off_diag_indx;
                values[i_offdiag] = propensity(state, reaction);

                col_indices[i_diag] = (int) tid;
                col_indices[i_offdiag] = off_diag_indx;
            } else {
                state = states + dim * tid;
                values[rowptr] = propensity(state, reaction);
                values[rowptr] *= -1.0;
                col_indices[rowptr] = (int) tid;
            }
        }
    }
}
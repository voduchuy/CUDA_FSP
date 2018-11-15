//
// Created by huy on 11/3/18.
//

#ifndef CUDA_FSP_FSP_MAT_KERNELS_H
#define CUDA_FSP_FSP_MAT_KERNELS_H
#include "cuda_runtime.h"
#include "cme_util.h"

namespace cuFSP{
    typedef double (*PropFun) (int* x, int reaction);
    __global__
    void
    fspmat_component_get_nnz_per_row(int *nnz_per_row, int *off_indx, int *states, int reaction, int n_rows,
                                     int n_species, int *fsp_bounds,
                                     int *stoich_vals, int *stoich_colidxs, int *stoich_rowptrs);

    __global__
    void
    fspmat_component_fill_data_csr(double *values, int *col_indices, int *row_ptrs, int n_rows, int reaction,
                                   int *off_diag_indices, int *states, int dim, PropFun propensity);
}
#endif //CUDA_FSP_FSP_MAT_KERNELS_H

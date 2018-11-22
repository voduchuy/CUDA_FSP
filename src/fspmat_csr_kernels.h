//
// Created by huy on 11/3/18.
//

#ifndef CUDA_FSP_FSP_MAT_KERNELS_H
#define CUDA_FSP_FSP_MAT_KERNELS_H
#include "cuda_runtime.h"
#include "cme_util.h"

namespace cuFSP{
    struct CUDACSRMatSet{
        cusparseHandle_t handle;
        cusparseMatDescr_t descr;
        std::vector<CSRMat> term;
        double h_one = 1.0;
        __host__ void action(double *x, double *y, double *coefs);
        __host__ void destroy();
    };

    __global__
    void
    fspmat_csr_get_nnz_per_row(int *nnz_per_row, int *off_indx, int *states, int reaction, int n_rows,
                               int n_species, int *fsp_bounds,
                               int *stoich_vals, int *stoich_colidxs, int *stoich_rowptrs);

    __global__
    void
    fspmat_csr_fill_data(double *values, int *col_indices, int *row_ptrs, int n_rows, int reaction,
                         int *off_diag_indices, int *states, int dim, PropFun propensity);

    __host__
    void generate_fsp_mats_cuda_csr(int *states, int n_states, int n_reactions, int n_species, int *fsp_dim,
                                    CSRMatInt stoich, PropFun prop_func, CUDACSRMatSet *csr);
    }
#endif //CUDA_FSP_FSP_MAT_KERNELS_H

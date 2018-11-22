//
// Created by Huy Vo on 11/21/18.
//

#ifndef CUDA_FSP_FSPMAT_HYB_KERNELS_H
#define CUDA_FSP_FSPMAT_HYB_KERNELS_H

#include <cuda_runtime.h>
#include "cme_util.h"

namespace cuFSP{
    struct HYBMatSet{
        int num_matrices;
        int n_rows;
        double* diag_vals;
        double* offdiag_vals;
        int* offdiag_colidxs;
        double *d_coefs;

        __host__ void action(double *x, double *y, double *coefs);
        __host__ void destroy();
    };

    __global__
    void fspmat_hyb_fill_data(int n_species, int n_reactions, int n_states, int *fsp_bounds, int *states, cuFSP::CSRMatInt stoich,
                              cuFSP::PropFun propensity,
                              double *diag_vals, double *offdiag_vals, int *offdiag_colindxs);

    __global__
    void fspmat_hyb_mv(int n_states, int n_reactions, double *diag_vals, double *offdiag_vals, int *offdiag_colidxs,
                       double *coef, double *x, double *y);

    __host__
    void generate_fsp_mats_hyb(int *states, int n_states, int n_reactions, int n_species, int *fsp_bounds,
                               CSRMatInt stoich, PropFun prop_func, HYBMatSet *hyb);
}

#endif //CUDA_FSP_FSPMAT_HYB_KERNELS_H

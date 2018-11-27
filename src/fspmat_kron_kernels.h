//
// Created by Huy Vo on 11/24/18.
//
#pragma once
#ifndef CUDA_FSP_FSPMAT_KRON_KERNELS_H
#define CUDA_FSP_FSPMAT_KRON_KERNELS_H

#include "cme_util.h"

namespace cuFSP {


    typedef double (*PropFactorFun)(int state, int species, int reaction);

    struct SDKronMatSet {
        int num_matrices; // Number of matrices in the set
        int num_factors; // Number of Kronecker factors for each matrix in the set
        int mat_data_size; // Number of elements in the data array for each matrix in the set
        int n_global; // Number of rows in each matrix
        int *n_bounds; // Bounds on each dimension
        double *vals; // Values of the shifted diagonal Kronecker factors of all matrices
        int *offsets; // Offsets of the shifted diagonal Kronecker factors
        double *d_coefs;
        int mv_sm_size;

        void action(double *x, double *y, double *coefs);
        void destroy();
    };


    __global__
    void fspmat_sdkron_mv(int n_species, int n_reactions, int n_global, int *n_bounds, double *vals, int *offsets,
                     const double *coefs, const double *x, double *y);

    __host__
    void
    fspmat_sdkron_fill_host_data(int n_species, int *fsp_bounds, int n_reactions, CSRMatInt stoich, PropFactorFun pffun,
                                 double *vals, int *offsets);

    __host__
    void generate_fsp_mats_sdkron(int n_reactions, int n_species, int *fsp_bounds, CSRMatInt stoich, PropFactorFun pffun,
                                      SDKronMatSet *sdkmatset);
}

#endif //CUDA_FSP_FSPMAT_KRON_KERNELS_H

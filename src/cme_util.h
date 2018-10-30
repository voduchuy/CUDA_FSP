#pragma once
#include <cuda_runtime.h>
#include <armadillo>
using namespace arma;
namespace cuFSP{
    struct cuda_csr_mat {
        double *vals = nullptr;
        int *col_idxs = nullptr;
        int *row_ptrs = nullptr;
        size_t n_rows, n_cols;
    };

    struct cuda_csr_mat_int {
        int *vals = nullptr;
        int *col_idxs = nullptr;
        int *row_ptrs = nullptr;
        size_t n_rows, n_cols;
    };
    __host__ __device__
    void indx2state(size_t indx, int *state, size_t dim, size_t *fsp_bounds);

    __host__ __device__
    void state2indx(int *state, int &indx, size_t dim, size_t *fsp_bounds);
}


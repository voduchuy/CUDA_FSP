#pragma once
#include <cuda_runtime.h>

#define CUDACHKERR() { \
cudaError_t ierr = cudaGetLastError();\
if (ierr != cudaSuccess){ \
    printf("%s in %s at line %d\n", cudaGetErrorString(ierr), __FILE__, __LINE__);\
    exit(EXIT_FAILURE); \
}\
}\

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
        size_t n_rows;
        size_t n_cols;
    };
    __device__
    void indx2state(size_t indx, int *state, size_t dim, size_t *fsp_bounds);

    __device__
    int state2indx(int *state, size_t dim, size_t *fsp_bounds);
}


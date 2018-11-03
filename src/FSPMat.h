#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <nvfunctional>
#include <thrust/scan.h>
#include <cusparse.h>
#include <armadillo>
#include "cme_util.h"
#include "thrust/device_vector.h"

using namespace arma;

namespace cuFSP{
    typedef std::function< Col<double> (double) > TcoefFun ;
    typedef double (*PropFun) (int* x, int reaction);
    typedef thrust::device_vector<double> thrust_dvec;

    __global__
    void fsp_get_states(int *d_states, size_t dim, size_t n_states, size_t *n_bounds);

    __global__
    void
    fspmat_component_get_nnz_per_row(int *nnz_per_row, int *off_indx, int* states, int reaction, size_t n_rows,
                                  size_t n_species, size_t *fsp_bounds,
                                  int *stoich_vals, int *stoich_colidxs, int *stoich_rowptrs);

    __global__
    void
    fspmat_component_fill_data_csr(double* values, int* col_indices, int* row_ptrs, size_t n_rows, int reaction,
                                int* off_diag_indices, int* states, size_t dim, PropFun propensity);

    class FSPMat
    {
        cusparseHandle_t  cusparse_handle;
        cusparseMatDescr_t cusparse_descr;

        size_t nst = 0,     // number of states
                ns = 0,      // number of species
                nr = 0;      // number of reactions

        std::vector<cuda_csr_mat>  term;

        double t = 0;

        TcoefFun tcoeffunc = nullptr;
        Col<double> tcoef;

        void destroy();
    public:
        // Functions to get member variables
        size_t get_n_rows();
        size_t get_n_species();
        size_t get_n_reactions();
        cuda_csr_mat* get_term(size_t i);

        // Constructor
        explicit FSPMat
        (cusparseHandle_t _handle,
                int *states, size_t n_states, size_t n_reactions, size_t n_species, size_t *fsp_dim,
                cuda_csr_mat_int stoich, TcoefFun t_func, PropFun prop_func);

        // Multiplication with a column vector
        void action (double t, thrust_dvec& x, thrust_dvec& y);

        // Destructor
        ~FSPMat();
    };
}

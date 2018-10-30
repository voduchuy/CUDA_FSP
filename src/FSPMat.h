#pragma once
#include <armadillo>   // Linear Algebra library with similar interface to MATLAB
#include <vector>
#include <cuda_runtime.h>
#include <nvfunctional>
#include <thrust/scan.h>
#include <cusparse.h>
#include "Model.h"
#include "cme_util.h"


using namespace arma;

namespace cuFSP{
    typedef std::function< Row<double> (double) > TcoefFun ;
    typedef nvstd::function< double (int*, int) > PropFun;

    __global__
    void fsp_get_states(int *d_states, size_t dim, size_t n_states, size_t *n_bounds);

    __global__
    void
    fspmat_component_get_nnz_per_row(int *nnz_per_row, int *off_indx, int reaction, size_t n_rows,
                                  size_t n_species, int* states, size_t *fsp_bounds, cuda_csr_mat_int stoich);

    __global__
    void
    fspmat_component_fill_data_csr(double* values, int* col_indices, int* row_ptrs, size_t n_rows, int reaction,
                                int* off_diag_indices, int* states, size_t dim, PropFun propensity);

    class FSPMat
    {
        cusparseHandle_t  cusparse_handle;
        cudaStream_t stream;
        cusparseMatDescr_t cusparse_descr;

        size_t nst = 0,     // number of states
                ns = 0,      // number of species
                nr = 0;      // number of reactions

        std::vector<cuda_csr_mat>  term;

        double t = 0;

        TcoefFun tcoeffunc = nullptr;
        Row<double> tcoef;
    public:
        // Functions to get member variables
        size_t get_n_rows();
        size_t get_n_species();
        size_t get_n_reactions();
        cuda_csr_mat* get_term(size_t i);

        // Constructor
        explicit FSPMat(int *states, size_t n_states, size_t n_reactions, size_t n_species, size_t *fsp_dim,
                cuda_csr_mat_int stoich, TcoefFun t_func, PropFun prop_func);

        // Multiplication with a column vector
        void action (double t, double *x, double *y);

        // Destructor
        ~FSPMat();
    };
}

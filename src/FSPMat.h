#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <nvfunctional>
#include <thrust/scan.h>
#include <cusparse.h>
#include <armadillo>
#include "cme_util.h"
#include "fsp_mat_kernels.h"
#include "thrust/device_vector.h"

namespace cuFSP{
    typedef std::function< void (double, double*) > TcoefFun ;
    enum precision {SINGLE, DOUBLE};
    enum matrix_format {CSR, HYB, KRONECKER};
    class FSPMat
    {
        cusparseHandle_t  cusparse_handle;
        cusparseMatDescr_t cusparse_descr;

        int nst = 0,     // number of states
                ns = 0,      // number of species
                nr = 0;      // number of reactions

        std::vector<cuda_csr_mat>  term;

        double t = 0;

        TcoefFun tcoeffunc = nullptr;
        double *tcoef = nullptr;

        void destroy();

        double *h_one = nullptr;
    public:
        // Functions to get member variables
        int get_n_rows();
        int get_n_species();
        int get_n_reactions();
        cuda_csr_mat* get_term(int i);

        // Constructor
        explicit FSPMat
                (int *states, int n_states, int n_reactions, int n_species, int *fsp_dim,
                 cuda_csr_mat_int stoich, TcoefFun t_func, PropFun prop_func);

        // Multiplication with a column vector
        void action(double t, double* x, double* y);
        // Destructor
        ~FSPMat();
    };
}

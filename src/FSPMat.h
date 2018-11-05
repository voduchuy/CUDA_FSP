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

using namespace arma;

namespace cuFSP{
    typedef std::function< Col<double> (double) > TcoefFun ;

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

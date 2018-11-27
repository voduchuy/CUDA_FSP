#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <nvfunctional>
#include <thrust/scan.h>
#include <cusparse.h>
#include <armadillo>
#include <thrust/device_vector.h>
#include "cme_util.h"
#include "fspmat_csr_kernels.h"
#include "fspmat_hyb_kernels.h"
#include "fspmat_kron_kernels.h"

namespace cuFSP {
    typedef std::function<void(double, double *)> TcoefFun;
    enum MatrixFormat {
        CUDA_CSR, HYB, KRONECKER
    };

    class FSPMat {
        MatrixFormat matrix_format;

        int nst = 0,     // number of states
                ns = 0,      // number of species
                nr = 0;      // number of reactions

        void* data_ptr;
        std::function<void(double *, double *, double *)> mv_ptr;

        double t = 0;
        TcoefFun tcoeffunc = nullptr;
        double *tcoef = nullptr;

        void destroy();
    public:
        // Functions to get member variables
        int get_n_rows();
        int get_n_species();
        int get_n_reactions();

        // Constructor
        explicit FSPMat
                (int n_reactions, int n_species, int *fsp_dim,
                 CSRMatInt stoich, TcoefFun t_func, PropFun prop_func, MatrixFormat format = CUDA_CSR);

        explicit FSPMat
                (int n_reactions, int n_species, int *fsp_dim,
                 CSRMatInt stoich, TcoefFun t_func, PropFactorFun pffunc, MatrixFormat format);

        // Multiplication with a column vector
        void action(double t, double *x, double *y);

        // Destructor
        ~FSPMat();
    };
}

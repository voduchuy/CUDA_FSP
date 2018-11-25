#include "FSPMat.h"
#include "driver_types.h"

using namespace arma;

namespace cuFSP {
    // Constructor
    // Precondition:
    // stoich stores the stoichiometry matrix, assumed to be in CSR format, with each row for one reaction
    FSPMat::FSPMat
            (int *states, int n_states, int n_reactions, int n_species, int *fsp_dim,
             CSRMatInt stoich, TcoefFun t_func, PropFun prop_func, MatrixFormat format) {

        matrix_format = format;
        // Initialize dimensions
        nst = n_states;
        nr = n_reactions;
        ns = n_species;
        tcoeffunc = t_func;

        cudaMallocHost((void **) &tcoef, nr * sizeof(double));
        CUDACHKERR();

        // Generate format-specific matrix data
        switch (matrix_format) {
            case CUDA_CSR:
                data_ptr = new CUDACSRMatSet;
                generate_fsp_mats_cuda_csr(states, n_states, n_reactions, n_species, fsp_dim, stoich, prop_func,
                                           (CUDACSRMatSet *) data_ptr);
                mv_ptr = [this](double *x, double *y, double *coefs) {
                    ((CUDACSRMatSet *) data_ptr)->action(x, y, coefs);
                };
                break;
            case HYB:
                data_ptr = new HYBMatSet;
                generate_fsp_mats_hyb(states, n_states, n_reactions, n_species, fsp_dim, stoich, prop_func,
                                      (HYBMatSet *) data_ptr);
                mv_ptr = [this](double *x, double *y, double *coefs) {
                    ((HYBMatSet *) data_ptr)->action(x, y, coefs);
                };
                break;
            default:
                throw std::runtime_error("FSPMat::FSPMat : requested format is currently not supported.");
        }
    }

    // Precondition:
    // stoich stores the stoichiometry matrix, assumed to be in CSR format, with each row for one reaction
    FSPMat::FSPMat
            (int *states, int n_states, int n_reactions, int n_species, int *fsp_dim,
             CSRMatInt stoich, TcoefFun t_func, PropFactorFun pffunc, MatrixFormat format) {

        matrix_format = format;
        // Initialize dimensions
        nst = n_states;
        nr = n_reactions;
        ns = n_species;
        tcoeffunc = t_func;

        cudaMallocHost((void **) &tcoef, nr * sizeof(double));
        CUDACHKERR();

        // Generate format-specific matrix data
        switch (matrix_format) {
            case KRONECKER:
                data_ptr = new SDKronMatSet;
                generate_fsp_mats_sdkron(states, n_states, n_reactions, n_species, fsp_dim, stoich, pffunc,
                                      (SDKronMatSet *) data_ptr);
                mv_ptr = [this](double *x, double *y, double *coefs) {
                    ((SDKronMatSet *) data_ptr)->action(x, y, coefs);
                };
                break;
            default:
                throw std::runtime_error("FSPMat::FSPMat : requested format is currently not supported.");
        }
    }

    // Destructor
    void FSPMat::destroy() {
        if (tcoef) {
            cudaFreeHost(tcoef);
        }
        switch (matrix_format){
            case CUDA_CSR:
                ((CUDACSRMatSet*) data_ptr)->destroy();
                delete (CUDACSRMatSet*) data_ptr;
                break;
            case HYB:
                ((HYBMatSet*) data_ptr)->destroy();
                delete (HYBMatSet*) data_ptr;
                break;
            default:
                break;
        }
    }

    FSPMat::~FSPMat() {
        destroy();
    }

    // Multiplication with a vector
    void FSPMat::action(double t, double *x, double *y) {
        tcoeffunc(t, tcoef);
        mv_ptr(x, y, tcoef);
    }

    // Getters
    int FSPMat::get_n_rows() {
        return nst;
    }

    int FSPMat::get_n_species() {
        return ns;
    }

    int FSPMat::get_n_reactions() {
        return nr;
    }
}
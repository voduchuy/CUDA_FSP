#pragma once

#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <vector>
#include <algorithm>
#include <armadillo>
#include "cme_util.h"

namespace cuFSP {
    using namespace arma;
    using MVFun = std::function<void(double *x, double *y)>;

/**
 * @brief Wrapper class for Krylov-based evaluation of the matrix exponential.
 * @details Compute the expression exp(t_f*A)*v, where A and v are cuFSP matrix and vector objects. The matrix does not enter explicitly but through the matrix-vector product. The Krylov-based approximation used is based on Sidje's Expokit, with the option to use Incomplete Orthogonalization Process in place of the full Arnoldi.
 */
    class KryExpvFSP {

    protected:
        cublasHandle_t cublas_handle;

        size_t n;   ///< Length of the solution vector

        MVFun matvec;
        thrust_dvec &sol_vec;

        double t_final;       ///< Final time/scaling of the matrix.

        size_t i_step = 0;

        double t_now;
        double t_new;
        double t_step;

        const double delta = 1.2;       ///< Safety factor in stepsize adaptivity
        const double gamma = 0.9;       ///< Safety factor in stepsize adaptivity

        double btol;

        double *wsp;               ///< workspace for the Krylov algorithm, need at least n*(m+2) elements

        arma::Mat<double> H;       ///< The small, dense Hessenberg matrix

        arma::Mat<double> F;

        bool vectors_created = false;

        double tol, anorm;

        std::vector<double*> V; ///< pointer to the workspace
    public:

        bool IOP = false;         ///< Flag for using incomplete orthogonalization. (default false)
        size_t q_iop = 2;         ///< IOP parameter, the current Krylov vector will be orthogonalized against q_iop-1 previous ones

        size_t m = 30;         ///< Size of the Krylov subspace for each step
        size_t max_nstep = 10000;
        size_t max_reject = 1000;


        /**
        * @brief Constructor for KExpv with vector data structures.
        */
        KryExpvFSP(double _t_final, MVFun &_matvec, thrust_dvec &_v, size_t _m, double _tol = 1.0e-8,
                   bool _iop = false, size_t _q_iop = 2, double _anorm = 1.0);

        /**
        * @brief Set the current time to 0.
        * @details The current solution vector will be kept, so solve() will integrate a new linear system with the current solution vector as the initial solution.
        */
        void reset_time() {
            t_now = 0.0;
        }


        /**
        * @brief Integrate all the way to t_final.
        */
        void solve();

        /**
        * @brief Advance to the furthest time possible using a Krylov basis of max dimension m.
        */
        void step();

        /**
        * @brief Check if the final time has been reached.
        */
        bool final_time_reached() {
            return (t_now >= t_final);
        }

        /**
         * @brief Destructor
         */
        ~KryExpvFSP();
    };
}

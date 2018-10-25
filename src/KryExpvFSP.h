#pragma once
#include <vector>
#include <algorithm>
#include <armadillo>

namespace cme {
    using namespace arma;
    using MVFun = std::function<Col<double> (Col<double>& x)>;

/**
 * @brief Wrapper class for Krylov-based evaluation of the matrix exponential.
 * @details Compute the expression exp(t_f*A)*v, where A and v are PETSC matrix and vector objects. The matrix does not enter explicitly but through the matrix-vector product. The Krylov-based approximation used is based on Sidje's Expokit, with the option to use Incomplete Orthogonalization Process in place of the full Arnoldi.
 */
    class KryExpvFSP {

    protected:

        MVFun matvec;
        Col<double>& sol_vec;

        double t_final;       ///< Final time/scaling of the matrix.

        size_t i_step = 0;

        double t_now;
        double t_new;
        double t_step;

        const double delta = 1.2;       ///< Safety factor in stepsize adaptivity
        const double gamma = 0.9;       ///< Safety factor in stepsize adaptivity

        double btol;

        std::vector<Col<double>> V;       ///< Container for the Krylov vectors
        arma::Mat<double> H;       ///< The small, dense Hessenberg matrix, each process in comm owns a copy of this matrix
        Col<double> av;
        arma::Mat<double> F;

        bool vectors_created = false;

    public:
        double tol = 1.0e-8;
        double anorm = 1.0;

        bool IOP = false;         ///< Flag for using incomplete orthogonalization. (default false)
        size_t q_iop = 2;         ///< IOP parameter, the current Krylov vector will be orthogonalized against q_iop-1 previous ones

        size_t m = 30;         ///< Size of the Krylov subspace for each step
        size_t max_nstep = 10000;
        size_t max_reject = 1000;


        /**
 * @brief Constructor for KExpv without initializing vector data structures
 * @details User must manually call update_vectors method before using the object.
 */
        KryExpvFSP(Col<double>& _v, double _t_final, size_t _m, double _tol = 1.0e-8, bool _iop = false, size_t _q_iop = 2, double _anorm = 1.0) :
                t_final(_t_final),
                m(_m),
                tol(_tol),
                IOP(_iop),
                q_iop(_q_iop),
                anorm(_anorm),
                t_now(0.0),
                sol_vec(_v)
        {
        }

/**
 * @brief Constructor for KExpv with vector data structures.
 */
        KryExpvFSP(double _t_final, MVFun& _matvec, Col<double>& _v, size_t _m, double _tol = 1.0e-8, bool _iop = false, size_t _q_iop = 2, double _anorm = 1.0) :
                t_final(_t_final),
                m(_m),
                tol(_tol),
                IOP(_iop),
                q_iop(_q_iop),
                anorm(_anorm),
                t_now(0.0),
                sol_vec(_v)
        {
            update_vectors(_v, _matvec);
        }

/**
 * @brief Set the current time to 0.
 * @details The current solution vector will be kept, so solve() will integrate a new linear system with the current solution vector as the initial solution.
 */
        void reset_time()
        {
            t_now = 0.0;
        }

/**
 * @brief Extend the final time.
 */
        void update_final_time(double _t_final)
        {
            t_final = _t_final;
        }

/**
 * @brief Set a new initial solution and reset the current time to 0.
 * @details The new initial solution must have the same distributed structure as the old solution.
 */
        void reset_init_sol(Col<double>& new_initial_solution)
        {
            t_now = 0.0;
            sol_vec = new_initial_solution;
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
        bool final_time_reached()
        {
            return (t_now >= t_final);
        }

/**
 * @brief Update the vector data structure
 */
        void update_vectors(Col<double>& _v, MVFun& _matvec)
        {
            matvec = _matvec;
            sol_vec = _v;
            V.resize(m+1);
        }


    };
}

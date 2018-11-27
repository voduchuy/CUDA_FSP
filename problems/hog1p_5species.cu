//
// Created by Huy Vo on 11/26/18.
//
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cvode/cvode.h>
#include <nvector/nvector_cuda.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_spbcgs.h>
#include <cvode/cvode_spils.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>

#include "cme_util.h"
#include "FSPMat.h"

namespace hog1p {
// reaction parameters
    const double k12{1.29}, k21{1.0e0}, k23{0.0067},
            k32{0.027}, k34{0.133}, k43{0.0381},
            kr2{0.0116}, kr3{0.987}, kr4{0.0538},
            trans{0.01}, gamma{0.0049},
// parameters for the time-dependent factors
            r1{6.9e-5}, r2{7.1e-3}, eta{3.1}, Ahog{9.3e09}, Mhog{6.4e-4};

// propensity function
    __device__
    double propensity(int *X, int k) {
        switch (X[0]) {
            case 0: {
                switch (k) {
                    case 0:
                        return k12;
                    case 1:
                        return 0.0;
                    case 2:
                        return 0.0;
                    case 3:
                        return 0.0;
                    case 4:
                        return 0.0;
                }
            }
            case 1: {
                switch (k) {
                    case 0:
                        return k23;
                    case 1:
                        return 0.0;
                    case 2:
                        return k21;
                    case 3:
                        return kr2;
                    case 4:
                        return kr2;
                }
            }
            case 2: {
                switch (k) {
                    case 0:
                        return k34;
                    case 1:
                        return k32;
                    case 2:
                        return 0.0;
                    case 3:
                        return kr3;
                    case 4:
                        return kr3;
                }
            }
            case 3: {
                switch (k) {
                    case 0:
                        return 0.0;
                    case 1:
                        return k43;
                    case 2:
                        return 0.0;
                    case 3:
                        return kr4;
                    case 4:
                        return kr4;
                }
            }
        }

        switch (k) {
            case 5:
                return trans * double(X[1]);
            case 6:
                return trans * double(X[2]);
            case 7:
                return gamma * double(X[3]);
            case 8:
                return gamma * double(X[4]);
        }

        return 0.0;
    }

    __host__
    double propensity_factor(int X, int species, int reaction) {
        if (species == 0) {
            switch (X){
                case 0: {
                    switch (reaction) {
                        case 0:
                            return k12;
                        case 1:
                            return 0.0;
                        case 2:
                            return 0.0;
                        case 3:
                            return 0.0;
                        case 4:
                            return 0.0;
                    }
                }
                case 1: {
                    switch (reaction) {
                        case 0:
                            return k23;
                        case 1:
                            return 0.0;
                        case 2:
                            return k21;
                        case 3:
                            return kr2;
                        case 4:
                            return kr2;
                    }
                }
                case 2: {
                    switch (reaction) {
                        case 0:
                            return k34;
                        case 1:
                            return k32;
                        case 2:
                            return 0.0;
                        case 3:
                            return kr3;
                        case 4:
                            return kr3;
                    }
                }
                case 3: {
                    switch (reaction) {
                        case 0:
                            return 0.0;
                        case 1:
                            return k43;
                        case 2:
                            return 0.0;
                        case 3:
                            return kr4;
                        case 4:
                            return kr4;
                    }
                }
                default:
                    return 1.0;
            }
        }

        switch (reaction) {
            case 5:
                if (species == 1) return trans * double(X);
            case 6:
                if (species == 2) return trans * double(X);
            case 7:
                if (species == 3) return gamma * double(X);
            case 8:
                if (species == 4) return gamma * double(X);
            default:
                return 1.0;
        }
        return 1.0;
    }

// function to compute the time-dependent coefficients of the propensity functions
    void t_func(double t, double *out) {
        for (int i = 0; i < 9; ++i) {
            out[i] = 1.0;
        }

        double h1 = (1.0 - exp(-r1 * t)) * exp(-r2 * t);

        double hog1p = pow(h1 / (1.0 + h1 / Mhog), eta) * Ahog;

        out[2] = std::max(0.0, 3200.0 - 7710.0 * (hog1p));
        //u(2) = std::max(0.0, 3200.0 - (hog1p));
    }
}

/* RHS of CME routine. */
__host__
static int cvode_rhs(double t, N_Vector u, N_Vector udot, void *FSPMat_ptr) {
    double *udata = N_VGetDeviceArrayPointer_Cuda(u);
    double *udotdata = N_VGetDeviceArrayPointer_Cuda(udot);
    thrust::fill(thrust::device_pointer_cast<double>(udotdata),
                 thrust::device_pointer_cast<double>(udotdata + ((cuFSP::FSPMat *) FSPMat_ptr)->get_n_rows()), 0.0);
    ((cuFSP::FSPMat *) FSPMat_ptr)->action(t, udata, udotdata);
    CUDACHKERR();
    return 0;
}

__device__ cuFSP::PropFun prop_pointer = &hog1p::propensity;

/* Jacobian-times-vector routine. */
__host__
static int cvode_jac(N_Vector v, N_Vector Jv, realtype t,
                     N_Vector u, N_Vector fu,
                     void *FSPMat_ptr, N_Vector tmp) {
    double *vdata = N_VGetDeviceArrayPointer_Cuda(v);
    double *Jvdata = N_VGetDeviceArrayPointer_Cuda(Jv);
    thrust::fill(thrust::device_pointer_cast<double>(Jvdata),
                 thrust::device_pointer_cast<double>(Jvdata + ((cuFSP::FSPMat *) FSPMat_ptr)->get_n_rows()), 0.0);
    ((cuFSP::FSPMat *) FSPMat_ptr)->action(t, vdata, Jvdata);
    CUDACHKERR();
    return 0;
}

static int check_flag(void *flagvalue, const char *funcname, int opt);

int main() {
    int n_species = 5;
    int n_reactions = 9;
    double t_final = 5.0 * 60;

    double rel_tol = 1.0e-2, abs_tol = 1.0e-8;

    int flag;

    int stoich_vals[] = {1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1};
    int stoich_colidxs[] = {0, 0, 0, 1, 2, 1, 3, 2, 4, 3, 4};
    int stoich_rowptrs[] = {0, 1, 2, 3, 4, 5, 7, 9, 10, 11};

    // stoichiometric matrix of the toggle switch model
//    const arma::Mat<int> SM {
//            {  1,  -1,  -1, 0, 0,  0,  0,  0,  0 },
//            {  0,  0,   0,  1, 0, -1,  0,  0,  0 },
//            {  0,  0,   0,  0, 1,  0, -1,  0,  0 },
//            {  0,  0,   0,  0, 0,  1,  0, -1,  0 },
//            {  0,  0,   0,  0, 0,  0,  1,  0, -1 },
//    };

    cuFSP::CSRMatInt stoich;
    stoich.vals = &stoich_vals[0];
    stoich.col_idxs = &stoich_colidxs[0];
    stoich.row_ptrs = &stoich_rowptrs[0];
    stoich.n_rows = n_reactions;
    stoich.n_cols = n_species;
    stoich.nnz = 11;

    int n_bounds[] = {3, 50, 50, 60, 60};
    int n_states = cuFSP::rect_fsp_num_states(n_species, n_bounds);
    std::cout << "Total number of states:" << n_states << "\n";


    cuFSP::PropFun host_prop_ptr;
    cudaMemcpyFromSymbol(&host_prop_ptr, prop_pointer, sizeof(cuFSP::PropFun));
    CUDACHKERR();
    cuFSP::FSPMat A(n_reactions, n_species, n_bounds, stoich, &hog1p::t_func, host_prop_ptr, cuFSP::HYB);
//    cuFSP::FSPMat A
//            (n_reactions, n_species, n_bounds,
//             stoich, &hog1p::t_func, &hog1p::propensity_factor, cuFSP::KRONECKER);

    /* Create a CUDA vector with initial values */
    N_Vector p0 = N_VNew_Cuda(n_states);  /* Allocate p0 vector */
    if (check_flag((void *) p0, "N_VNew_Cuda", 0)) return (1);
    double *p0_h = N_VGetHostArrayPointer_Cuda(p0);
    for (int i = 0; i < n_states; ++i) {
        p0_h[i] = 0.0;
    }
    p0_h[0] = 1.0;
    N_VCopyToDevice_Cuda(p0);

    /* Call CVodeCreate to create the solver memory and specify the
    * Backward Differentiation Formula and the use of a Newton iteration */
    void *cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
    if (check_flag((void *) cvode_mem, "CVodeCreate", 0)) return (1);

    /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in u'=f(t,u), the initial time T0, and
    * the initial dependent variable vector u. */
    flag = CVodeInit(cvode_mem, cvode_rhs, 0.0, p0);
    if (check_flag(&flag, "CVodeInit", 1)) return (1);

    /* Call CVodeSStolerances to specify the scalar relative tolerance
* and scalar absolute tolerance */
    flag = CVodeSStolerances(cvode_mem, rel_tol, abs_tol);
    if (check_flag(&flag, "CVodeSStolerances", 1)) return (1);

    /* Set the pointer to user-defined data */
    flag = CVodeSetUserData(cvode_mem, (void *) &A);
    if (check_flag(&flag, "CVodeSetUserData", 1)) return (1);

    flag = CVodeSetMaxNumSteps(cvode_mem, 10000000);
    flag = CVodeSetMaxConvFails(cvode_mem, 10000000);
    flag = CVodeSetStabLimDet(cvode_mem, 1);
    flag = CVodeSetMaxNonlinIters(cvode_mem, 100000);

    /* Create SPGMR solver structure without preconditioning
    * and the maximum Krylov dimension maxl */
//    SUNLinearSolver LS =  SUNSPGMR(p0, PREC_NONE, 10);
//    if(check_flag(&flag, "SUNSPGMR", 1)) return(1);

    SUNLinearSolver LS = SUNSPBCGS(p0, PREC_NONE, 0);
    if (check_flag(&flag, "SUNSPBCGS", 1)) return (1);

    /* Set CVSpils linear solver to LS */
    flag = CVSpilsSetLinearSolver(cvode_mem, LS);
    if (check_flag(&flag, "CVSpilsSetLinearSolver", 1)) return (1);

    /* Set the JAcobian-times-vector function */
    flag = CVSpilsSetJacTimes(cvode_mem, NULL, cvode_jac);
    if (check_flag(&flag, "CVSpilsSetJacTimesVecFn", 1)) return (1);

    double t = 0.0;
    double psum = 0.0;
    double *p0_d = N_VGetDeviceArrayPointer_Cuda(p0);

    while (t < t_final) {
        flag = CVode(cvode_mem, t_final, p0, &t, CV_ONE_STEP);
        if (check_flag(&flag, "CVode", 1)) break;
        psum = thrust::reduce(thrust::device_pointer_cast<double>(p0_d),
                              thrust::device_pointer_cast<double>(p0_d + n_states));
        std::cout << "t = " << t << " psum = " << psum << "\n";
    }

    assert(std::abs(1.0 - psum) <= 1.0e-10);

    long num_step;
    flag = CVodeGetNumSteps(cvode_mem, &num_step);
    check_flag(&flag, "CVodeGetNumSteps", 1);
    std::cout << "CVODE takes " << num_step << " steps.\n";

    SUNLinSolFree(LS);
    CVodeFree(&cvode_mem);
    return 0;
}


/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */

static int check_flag(void *flagvalue, const char *funcname, int opt) {
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return (1);
    }

        /* Check if flag < 0 */

    else if (opt == 1) {
        errflag = (int *) flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return (1);
        }
    }

        /* Check if function returned NULL pointer - no memory allocated */

    else if (opt == 2 && flagvalue == NULL) {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return (1);
    }

    return (0);
}
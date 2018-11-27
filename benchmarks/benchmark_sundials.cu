//
// Created by Huy Vo on 10/30/18.
//
#include <armadillo>
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

/* Parameters for the propensity functions */
const double ayx{2.6e-3}, axy{6.1e-3}, nyx{3e0}, nxy{2.1e0},
        kx0{2.2e-3}, kx{1.7e-2}, dx{3.8e-4}, ky0{6.8e-5}, ky{1.6e-2}, dy{3.8e-4};

__device__ __host__
double toggle_propensity(int *x, int reaction) {
    double prop_val;
    switch (reaction) {
        case 0:
            prop_val = 1.0;
            break;
        case 1:
            prop_val = 1.0 / (1.0 + ayx*std::pow(1.0 * x[1], nyx));
            break;
        case 2:
            prop_val = 1.0 * x[0];
            break;
        case 3:
            prop_val = 1.0;
            break;
        case 4:
            prop_val = 1.0 / (1.0 + axy*std::pow(1.0 * x[0], nxy));
            break;
        case 5:
            prop_val = 1.0 * x[1];
            break;
    }
    return prop_val;
}

__device__ cuFSP::PropFun prop_pointer = &toggle_propensity;

__device__ __host__
void t_func(double t, double* out){
//    return {(1.0 + std::cos(t))*kx0, kx, dx, (1.0 + std::sin(t))*ky0, ky, dy};
    out[0] = kx0;
    out[1] = kx*(1.0 + cos(t/1000));
    out[2] = dx*exp(-t/3600.0);
    out[3] = ky0;
    out[4] = ky;
    out[5] = dy*exp(-t/3600.0);
}

/* RHS of CME routine. */
__host__
static int cvode_rhs(double t, N_Vector u, N_Vector udot, void* FSPMat_ptr){
    double* udata = N_VGetDeviceArrayPointer_Cuda(u);
    double* udotdata = N_VGetDeviceArrayPointer_Cuda(udot);
    thrust::fill(thrust::device_pointer_cast<double>(udotdata),
                 thrust::device_pointer_cast<double>(udotdata+ ((cuFSP::FSPMat*) FSPMat_ptr)->get_n_rows()), 0.0);
    ((cuFSP::FSPMat*) FSPMat_ptr)->action(t, udata, udotdata);
    CUDACHKERR();
    return 0;
}

/* Jacobian-times-vector routine. */
__host__
static int cvode_jac(N_Vector v, N_Vector Jv, realtype t,
               N_Vector u, N_Vector fu,
               void *FSPMat_ptr, N_Vector tmp)
{
    double* vdata = N_VGetDeviceArrayPointer_Cuda(v);
    double* Jvdata = N_VGetDeviceArrayPointer_Cuda(Jv);
    thrust::fill(thrust::device_pointer_cast<double>(Jvdata),
                 thrust::device_pointer_cast<double>(Jvdata+ ((cuFSP::FSPMat*) FSPMat_ptr)->get_n_rows()), 0.0);
    ((cuFSP::FSPMat*) FSPMat_ptr)->action(t, vdata, Jvdata);
    CUDACHKERR();
    return 0;
}

static int check_flag(void *flagvalue, const char *funcname, int opt);

int main()
{
    int n_species = 2;
    int n_reactions = 6;
    double t_final = 8*3600;

    double rel_tol = 1.0, abs_tol = 1.0e-8;

    int flag;

    int stoich_vals[] = {1, 1, -1, 1,1, -1};
    int stoich_colidxs[] = {0, 0, 0, 1, 1, 1};
    int stoich_rowptrs[] = {0, 1, 2, 3, 4, 5,6};

    cuFSP::CSRMatInt stoich;
    stoich.vals = &stoich_vals[0];
    stoich.col_idxs = &stoich_colidxs[0];
    stoich.row_ptrs = &stoich_rowptrs[0];
    stoich.n_rows = 6;
    stoich.n_cols = 2;
    stoich.nnz = 6;

    int n_bounds[] = {1<<10, 1<<10};
    std::cout << n_bounds[0] << " " << n_bounds[1] << "\n";
    int n_states = cuFSP::rect_fsp_num_states(n_species, n_bounds);
    std::cout << "Total number of states:" << n_states << "\n";

    cuFSP::PropFun host_prop_ptr;
    cudaMemcpyFromSymbol(&host_prop_ptr, prop_pointer, sizeof(cuFSP::PropFun)); CUDACHKERR();
    cuFSP::FSPMat A
    (n_reactions, n_species, n_bounds,
            stoich, &t_func, host_prop_ptr, cuFSP::HYB); CUDACHKERR();
    cudaDeviceSynchronize();


    /* Create a CUDA vector with initial values */
    N_Vector p0 = N_VNew_Cuda(n_states);  /* Allocate p0 vector */
    if(check_flag((void*)p0, "N_VNew_Cuda", 0)) return(1);
    double* p0_h = N_VGetHostArrayPointer_Cuda(p0);
    for (int i = 0; i < n_states; ++i){
        p0_h[i] = 0.0;
    }
    p0_h[0] = 1.0;
    N_VCopyToDevice_Cuda(p0);

    /* Call CVodeCreate to create the solver memory and specify the
    * Backward Differentiation Formula and the use of a Newton iteration */
    void *cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
    if(check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

    /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in u'=f(t,u), the initial time T0, and
    * the initial dependent variable vector u. */
    flag = CVodeInit(cvode_mem, cvode_rhs, 0.0, p0);
    if(check_flag(&flag, "CVodeInit", 1)) return(1);

    /* Call CVodeSStolerances to specify the scalar relative tolerance
* and scalar absolute tolerance */
    flag = CVodeSStolerances(cvode_mem, rel_tol, abs_tol);
    if (check_flag(&flag, "CVodeSStolerances", 1)) return(1);

    /* Set the pointer to user-defined data */
    flag = CVodeSetUserData(cvode_mem, (void*) &A);
    if(check_flag(&flag, "CVodeSetUserData", 1)) return(1);

    flag = CVodeSetMaxNumSteps(cvode_mem, 10000000);
    flag = CVodeSetMaxConvFails(cvode_mem, 10000000);
    flag = CVodeSetStabLimDet(cvode_mem, 1);
    flag = CVodeSetMaxNonlinIters(cvode_mem, 100000);

    /* Create SPGMR solver structure without preconditioning
    * and the maximum Krylov dimension maxl */
//    SUNLinearSolver LS =  SUNSPGMR(p0, PREC_NONE, 10);
//    if(check_flag(&flag, "SUNSPGMR", 1)) return(1);

    SUNLinearSolver LS = SUNSPBCGS(p0, PREC_NONE, 0);
    if(check_flag(&flag, "SUNSPBCGS", 1)) return(1);

    /* Set CVSpils linear solver to LS */
    flag = CVSpilsSetLinearSolver(cvode_mem, LS);
    if(check_flag(&flag, "CVSpilsSetLinearSolver", 1)) return(1);

    /* Set the JAcobian-times-vector function */
    flag = CVSpilsSetJacTimes(cvode_mem, NULL, cvode_jac);
    if(check_flag(&flag, "CVSpilsSetJacTimesVecFn", 1)) return(1);

    double t = 0.0;
    double psum = 0.0;
    double *p0_d = N_VGetDeviceArrayPointer_Cuda(p0);

    while (t < t_final){
        flag = CVode(cvode_mem, t_final, p0, &t, CV_ONE_STEP);
        if(check_flag(&flag, "CVode", 1)) break;
    }

    psum = thrust::reduce(thrust::device_pointer_cast<double>(p0_d), thrust::device_pointer_cast<double>(p0_d+n_states));
    std::cout << "t = " << t << " psum = " << psum << "\n";
    assert(std::abs(1.0 - psum) <= 1.0e-10);

    long num_step;
    flag = CVodeGetNumSteps(cvode_mem, &num_step);
    check_flag(&flag, "CVodeGetNumSteps", 1);
    std::cout << "CVODE takes " << num_step << " steps.\n";

    return 0;
}


/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */

static int check_flag(void *flagvalue, const char *funcname, int opt)
{
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return(1); }

        /* Check if flag < 0 */

    else if (opt == 1) {
        errflag = (int *) flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return(1); }}

        /* Check if function returned NULL pointer - no memory allocated */

    else if (opt == 2 && flagvalue == NULL) {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return(1); }

    return(0);
}
#include "KryExpvFSP.h"



namespace cuFSP {

    KryExpvFSP::KryExpvFSP(double _t_final, MVFun &_matvec, thrust_dvec &_v, size_t _m, double _tol,
                           bool _iop, size_t _q_iop, double _anorm):
                           t_final(_t_final),
                           matvec(_matvec),
                           sol_vec(_v),
                           m(_m),
                           tol(_tol),
                           IOP(_iop),
                           q_iop(_q_iop),
                           anorm(_anorm){
        n = _v.size();
        cudaMallocManaged(&wsp, (n*(m+2) + m+2)*sizeof(double));

        // Pointers to the Krylov vectors
        V.resize(m+2);
        V[0] = wsp;
        for (size_t i{1}; i < m+2; ++i)
        {
            V[i] = V[i-1] + n;
        }

        cublasStatus_t stat = cublasCreate_v2(&cublas_handle); CUBLASCHKERR(stat);
        CUDACHKERR();
    }

    void KryExpvFSP::step() {
        cublasStatus_t stat;

        double beta, s, avnorm, xm, err_loc;
        double zero = 0.0;

        stat = cublasDnrm2_v2(cublas_handle, (int) n, (double*) thrust::raw_pointer_cast(&sol_vec[0]), 1, &beta);
        CUDACHKERR();
        CUBLASCHKERR(stat);

        if (i_step == 0) {
            xm = 1.0 / double(m);
            //double anorm { norm(A, 1) };
            double fact = pow((m + 1) / exp(1.0), m + 1) * sqrt(2 * (3.1416) * (m + 1));
            t_new = (1.0 / anorm) * pow((fact * tol) / (4.0 * beta * anorm), xm);
            btol = anorm * tol; // tolerance for happy breakdown
        }

        size_t mx;
        size_t mb{m};
        size_t k1{2};

        double tau = std::min(t_final - t_now, t_new);
        H = arma::zeros(m + 2, m + 2);

        double betainv = 1.0/beta;
        stat = cublasDcopy_v2(cublas_handle, (int) n, (double*) thrust::raw_pointer_cast(&sol_vec[0]), 1, V[0], 1);
        CUBLASCHKERR(stat); CUDACHKERR();
        stat = cublasDscal_v2(cublas_handle, (int) n, &betainv, V[0], 1); CUBLASCHKERR(stat); CUDACHKERR();


        size_t istart = 0;
        /* Arnoldi loop */
        for (size_t j{0}; j < m; j++) {
            matvec(V[j], V[j+1]);

            /* Orthogonalization */
            if (IOP) istart = ( (int) j >= q_iop - 1) ? j - q_iop + 1 : 0;

            for (size_t i{istart}; i <= j; i++) {
                // H(i, j) =  dot(V[j + 1], V[i]);
                stat = cublasDdot_v2(cublas_handle, (int) n, V[j+1], 1, V[i], 1, &H(i,j)); CUBLASCHKERR(stat); CUDACHKERR();

                //V[j + 1] = V[j + 1] - H(i, j) * V[i];
                H(i,j) = -1.0*H(i,j);
                stat = cublasDaxpy(cublas_handle, (int) n, &H(i,j), V[i], 1, V[j+1], 1); CUBLASCHKERR(stat); CUDACHKERR();

                H(i,j) = -1.0*H(i,j);
            }
//            s = norm(V[j + 1], 2);
            stat = cublasDnrm2_v2(cublas_handle, n, V[j+1], 1, &s); CUBLASCHKERR(stat); CUDACHKERR();

            if (s < btol) {
                k1 = 0;
                mb = j + 1;
                tau = t_final - t_now;
#ifdef KEXPV_VERBOSE
                std::cout << "happy breakdown.\n";
#endif
                break;
            }

            H(j + 1, j) = s;
            double sinv = 1.0/s;
//            V[j + 1] = V[j + 1] / s;
            stat = cublasDscal_v2(cublas_handle, n, &sinv, V[j+1], 1);

            CUBLASCHKERR(stat); CUDACHKERR();
        }


        if (k1 != 0) {
            H(m + 1, m) = 1.0;
            matvec(V[mb], V[mb+1]);
            stat = cublasDnrm2_v2(cublas_handle, n, V[mb+1], 1, &avnorm); CUBLASCHKERR(stat); CUDACHKERR();
        }

        size_t ireject{0};
        while (ireject < max_reject) {
            mx = mb + k1;
            F = expmat(tau * H);
            if (k1 == 0) {
                err_loc = btol;
                break;
            } else {
                double phi1 = std::abs(beta * F(mx-2, 0));
                double phi2 = std::abs(beta * F(mx-1, 0) * avnorm);

                if (phi1 > phi2 * 10.0) {
                    err_loc = phi2;
                    xm = 1.0 / double(mx);
                } else if (phi1 > phi2) {
                    err_loc = (phi1 * phi2) / (phi1 - phi2);
                    xm = 1.0 / double(mx);
                } else {
                    err_loc = phi1;
                    xm = 1.0 / double(mx - 1);
                }
            }

            if (err_loc <= delta * tau * tol) {
                break;
            } else {
                tau = gamma * tau * pow(tau * tol / err_loc, xm);
                double s = pow(10.0, floor(log10(tau)) - 1);
                tau = ceil(tau / s) * s;
                if (ireject == max_reject) {
                    std::cout << "Maximum number of failed steps reached.";
                    t_now = t_final;
                    break;
                }
                ireject++;
            }
        }

        mx = mb + (size_t) std::max(0, (int) k1 - 1);

        double *F0 = V[m+1] + n;
        cudaMemcpy(F0, F.colptr(0), mx*sizeof(double), cudaMemcpyHostToDevice); CUDACHKERR();

        stat = cublasDgemv_v2(cublas_handle, CUBLAS_OP_N, (int) n, (int) mx, &beta, V[0], (int) n, F0, 1, &zero,
                              (double*) thrust::raw_pointer_cast(&sol_vec[0]), 1);
        CUBLASCHKERR(stat); CUDACHKERR();


        t_now = t_now + tau;
        t_new = gamma * tau * pow(tau * tol / err_loc, xm);
        s = pow(10.0, floor(log10(t_new)) - 1.0);
        t_new = ceil(t_new / s) * s;

#ifdef KEXPV_VERBOSE
//        std::cout << "t_now = " << t_now << " err_loc = " << err_loc << "\n";
        printf("i_step = %d \n t_now = %.2f err_loc = %.2e \n", i_step , t_now, err_loc);
#endif
        i_step++;

    }

    void KryExpvFSP::solve() {
        while (!final_time_reached()) {
            step();
        }
    }

    KryExpvFSP::~KryExpvFSP(){
        cudaFree(wsp); CUDACHKERR();
        cublasDestroy_v2(cublas_handle);
    }
}

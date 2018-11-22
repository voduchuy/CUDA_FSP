#include "KryExpvFSP.h"


namespace cuFSP {

    KryExpvFSP::KryExpvFSP(double _t_final, MVFun &_matvec, thrust_dvec &_v, int _m, double _tol,
                           bool _iop, int _q_iop, double _anorm) :
            t_final(_t_final),
            matvec(_matvec),
            sol_vec(_v),
            m(_m),
            tol(_tol),
            IOP(_iop),
            q_iop(_q_iop),
            anorm(_anorm) {
        n = (int) _v.size();
        cudaMalloc(&wsp, (n * (m + 2) + (m + 2) * (m + 2)) * sizeof(double));

        // Pointers to the Krylov vectors
        V.resize((size_t) m + 2);
        V[0] = wsp;
        for (size_t i{1}; i < m + 2; ++i) {
            V[i] = V[i - 1] + n;
        }

        cudaMallocHost((void **) &pinned_H, (m + 2) * (m + 2) * sizeof(double));
        cudaMallocHost((void **) &pinned_F, (m + 2) * (m + 2) * sizeof(double));

        // Arma wrapper for Hessenberg matrix on host
        H = arma::Mat<double>(pinned_H, (size_t) m + 2, (size_t) m + 2, false, true);
        F = arma::Mat<double>(pinned_F, (size_t) m + 2, (size_t) m + 2, false, true);

        cublasStatus_t stat = cublasCreate_v2(&cublas_handle);
        CUBLASCHKERR(stat);
        CUDACHKERR();
    }

    void KryExpvFSP::step() {
        cublasStatus_t stat;

        stat = cublasDnrm2_v2(cublas_handle, n, (double *) thrust::raw_pointer_cast(&sol_vec[0]), 1, &beta);
        CUDACHKERR();
        CUBLASCHKERR(stat);

        if (i_step == 0) {
            xm = 1.0 / double(m);
            //double anorm { norm(A, 1) };
            double fact = pow((m + 1) / exp(1.0), m + 1) * sqrt(2 * (3.1416) * (m + 1));
            t_new = (1.0 / anorm) * pow((fact * tol) / (4.0 * beta * anorm), xm);
            btol = anorm * tol; // tolerance for happy breakdown
        }

        mb = m;
        double tau = std::min(t_final - t_now, t_new);

        stat = cublasDscal_v2(cublas_handle, (n * (m + 2) + (m + 2) * (m + 2)) , &zero, wsp, 1);
        CUBLASCHKERR(stat);

        double betainv = 1.0 / beta;
        stat = cublasDcopy_v2(cublas_handle, n, (double *) thrust::raw_pointer_cast(&sol_vec[0]), 1, V[0], 1);
        CUBLASCHKERR(stat);
        CUDACHKERR();
        stat = cublasDscal_v2(cublas_handle, n, &betainv, V[0], 1);
        CUBLASCHKERR(stat);
        CUDACHKERR();

        int istart = 0;
        double *d_H = V[m + 1] + n;
        /* Arnoldi loop */
        for (int j{0}; j < m; j++) {
            matvec(V[j], V[j + 1]);

            /* Orthogonalization */
            if (IOP) istart = (j >= q_iop - 1) ? j - q_iop + 1 : 0;

            stat = cublasDgemv_v2(cublas_handle, CUBLAS_OP_T, n, (j - istart + 1), &one, V[istart], n, V[j + 1],
                                  1, &one, &d_H[istart + j * (m + 2)], 1);
            CUBLASCHKERR(stat);
            stat = cublasDgemv_v2(cublas_handle, CUBLAS_OP_N, n, (j - istart + 1), &minus_one, V[istart], n,
                                  &d_H[istart + j * (m + 2)],
                                  1, &one, V[j + 1], 1);
            CUBLASCHKERR(stat);
            CUDACHKERR();

            //            s = norm(V[j + 1], 2);
            stat = cublasDnrm2_v2(cublas_handle, n, V[j + 1], 1, &s);
            CUBLASCHKERR(stat);
            CUDACHKERR();

            if (s < btol) {
                k1 = 0;
                mb = j + 1;
                tau = t_final - t_now;
#ifdef KEXPV_VERBOSE
                std::cout << "happy breakdown.\n";
#endif
                break;
            }

            cudaMemcpy(&d_H[(j + 1) + j * (m + 2)], &s, sizeof(double), cudaMemcpyHostToDevice);
            CUDACHKERR();
//            H(j + 1, j) = s;
            double sinv = 1.0 / s;
//            V[j + 1] = V[j + 1] / s;
            stat = cublasDscal_v2(cublas_handle, n, &sinv, V[j + 1], 1);

            CUBLASCHKERR(stat);
            CUDACHKERR();
        }

        cudaMemcpy(H.colptr(0), d_H, (m + 2) * (m + 2) * sizeof(double), cudaMemcpyDeviceToHost);
        CUDACHKERR();

        if (k1 != 0) {
            H((size_t) m + 1, (size_t) m) = 1.0;
            matvec(V[mb], V[mb + 1]);
            stat = cublasDnrm2_v2(cublas_handle, n, V[mb + 1], 1, &avnorm);
            CUBLASCHKERR(stat);
            CUDACHKERR();
        }

        size_t ireject{0};
        while (ireject < max_reject) {
            mx = mb + k1;
            arma::expmat(F, tau * H);
            if (k1 == 0) {
                err_loc = btol;
                break;
            } else {
                double phi1 = std::abs(beta * F((size_t) mx - 2, 0));
                double phi2 = std::abs(beta * F((size_t) mx - 1, 0) * avnorm);

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

        mx = mb + std::max(0, k1 - 1);
        double *F0 = V[m + 1] + n;
        cudaMemcpy(F0, F.colptr(0), mx * sizeof(double), cudaMemcpyHostToDevice);
        CUDACHKERR();

        stat = cublasDgemv_v2(cublas_handle, CUBLAS_OP_N, n, mx, &beta, V[0], n, F0, 1, &zero,
                              (double *) thrust::raw_pointer_cast(&sol_vec[0]), 1);
        CUBLASCHKERR(stat);
        CUDACHKERR();


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

    KryExpvFSP::~KryExpvFSP() {
        cudaFree(wsp);
        CUDACHKERR();
        if (pinned_H) {
            cudaFreeHost(pinned_H);
            CUDACHKERR();
        }
        if (pinned_F) {
            cudaFreeHost(pinned_F);
            CUDACHKERR();
        }
        cublasDestroy_v2(cublas_handle);
    }
}

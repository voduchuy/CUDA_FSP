//
// Created by Huy Vo on 8/10/18.
//

#ifndef MCMC_FTT_CME_FSPSOLVER_H
#define MCMC_FTT_CME_FSPSOLVER_H

#include <armadillo>
#include "cme_util.h"
#include "FSPMat.h"
#include "KryExpvFSP.h"

namespace cme{
    using namespace arma;

    class FSPSolver{
        const struct Model& model;

        enum MatType {ELL, COO, CSR};

        FSPMat A;

        Col<double>& P;

        Row<double> times;
        size_t current_time_index {0};

        double fsp_tol;

        KryExpvFSP expv;

        std::function<Col<double> (Col<double>&)> matvec;

    public:

        FSPSolver(Col<double>& _p, const struct Model& _model, Row<double>& _times, double _fsp_tol,
                  double _mg_tol = 1.0e-8, double _kry_tol = 1.0e-8):
                model(_model),
                times(_times),
                fsp_tol(_fsp_tol),
                A(_model),
                expv(_p, _times(0), 30, _kry_tol, true),
                P(_p)
        {
            const Row<size_t>& FSPSize = model.FSPSize;
            const Mat<int>& states_init = model.states_init;
            const Col<double>& prob_init = model.probs_init;

            // Set up the initial vector
            if (P.n_elem < arma::prod(FSPSize+1))
            {
                P.resize(arma::prod(FSPSize+1));

                arma::Row<int> init_indices = sub2ind_fsp(FSPSize, states_init);

                for (size_t i{0}; i < init_indices.n_elem; ++i)
                {
                    P((uword) init_indices(i)) = prob_init(i);
                }
            }


            // Update the Magnus object with the initial vector information
            matvec = [this] (Col<double>& x){return A(0)*x;};
            expv.update_vectors(P, matvec);
        }

        FSPSolver(Col<double>& _p, const struct Model& _model, double _t_final, double _fsp_tol,
                  double _mg_tol = 1.0e-8, double _kry_tol = 1.0e-8):
                model(_model),
                times({_t_final}),
                fsp_tol(_fsp_tol),
                A(_model),
                expv(_p, _t_final, 30, _kry_tol, true),
                P(_p)
        {
            const Row<size_t>& FSPSize = model.FSPSize;
            const Mat<int>& states_init = model.states_init;
            const Col<double>& prob_init = model.probs_init;

            // Set up the initial vector
            if (P.n_elem < arma::prod(FSPSize+1))
            {
                P.resize(arma::prod(FSPSize+1));

                arma::Row<int> init_indices = sub2ind_fsp(FSPSize, states_init);

                for (size_t i{0}; i < init_indices.n_elem; ++i)
                {
                    P((uword) init_indices(i)) = prob_init(i);
                }
            }

            // Update the Magnus object with the initial vector information
            matvec = [this] (Col<double>& x){return A(0)*x;};
            expv.update_vectors(P, matvec);
        }

        void next_time();

        void solve();

        void reset_time()
        {
            current_time_index = 0;
            A(0.0);
            expv.reset_time();
            expv.update_final_time(times(0));
        }

    };
};




#endif //MCMC_FTT_CME_FSPSOLVER_H

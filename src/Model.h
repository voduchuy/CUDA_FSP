//
// Created by Huy Vo on 8/11/18.
//

#ifndef MCMC_FTT_CME_MODEL_H
#define MCMC_FTT_CME_MODEL_H

#include <armadillo>

namespace cme{
    using namespace arma;

    struct Model{
        typedef Mat<double> (* PropFun ) (Mat<int>);
        typedef Row<double> (* InputFun ) (double t, const Row<double>& parameters);

        Mat<int> states_init; // states are arranged column-wise
        Col<double> probs_init;

        Row<size_t> FSPSize;
        Mat<int> stoich_mat;
        PropFun propensity {nullptr};
        InputFun t_fun {nullptr};
        Row<double> parameters;
    };
}

#endif //MCMC_FTT_CME_MODEL_H

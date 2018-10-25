//
// Created by Huy Vo on 8/11/18.
//

#ifndef MCMC_FTT_CME_LIKELIHOOD_H
#define MCMC_FTT_CME_LIKELIHOOD_H

#include <cmath>
#include <armadillo>
#include <form.h>
#include "FSPSolver.h"

namespace cme{

    struct SingleCellData{
        Row<double> times;
        field<arma::Mat<int>> snapshots;
    };

    class Likelihood {
    protected:
        struct SingleCellData& data;
        struct Model& model;

        FSPSolver fsp;
        Col<double> full_dist{0.0};

        field<arma::Row<int>> data2fsp;
    public:
        Likelihood(struct SingleCellData& _data, struct Model& _model):
        data(_data),
        model(_model),
        full_dist({0.0}),
        fsp(full_dist, _model, _data.times, 1.0e-6)
        {
            data2fsp.set_size(data.times.n_elem);
            for (size_t it{0}; it < data.times.n_elem; ++it)
            {
                data2fsp(it) = sub2ind_fsp(model.FSPSize, data.snapshots(it));
            }

            full_dist.resize(prod(model.FSPSize+1));
            // Return the full distribution to the initial condition
            arma::Row<int> init_indices = sub2ind_fsp(model.FSPSize, model.states_init);

            for (size_t i{0}; i < init_indices.n_elem; ++i)
            {
                full_dist((uword) init_indices(i)) = model.probs_init(i);
            }

        };

        double evaluate(const Row<double>& parameters);

    };

    struct SingleCellData generate_data(struct Model& model, Row<double> times, Col<int> x0, size_t num_samples);
}



#endif //MCMC_FTT_CME_LIKELIHOOD_H

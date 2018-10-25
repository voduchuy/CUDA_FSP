//
// Created by Huy Vo on 8/11/18.
//

#include "Likelihood.h"

namespace cme{
    double Likelihood::evaluate(const Row<double> &parameters) {
        model.parameters = parameters;
        double ll = 0.0;

        fsp.reset_time();

        for(size_t it{0}; it < data.times.n_elem; ++it)
        {
            fsp.next_time();
            for (size_t j{0}; j < data2fsp(it).n_elem; ++j)
            {
                ll += std::log( std::max(1.0e-44, full_dist((uword) data2fsp(it)(j))) );
            }
        }

        // Return the full distribution to the initial condition
        full_dist.zeros();
        arma::Row<int> init_indices = sub2ind_fsp(model.FSPSize, model.states_init);

        for (size_t i{0}; i < init_indices.n_elem; ++i)
        {
            full_dist((uword) init_indices(i)) = model.probs_init(i);
        }

        return ll;
    };

    struct SingleCellData generate_data(struct Model& model, Row<double> times, Col<int> x0, size_t num_samples)
    {
        arma_rng::set_seed_random();

        field<Mat<int>> snapshots(times.n_elem);

        for (size_t it{0}; it < times.n_elem; ++it)
        {
            snapshots(it).set_size(x0.n_elem, num_samples);
            for (size_t i_sample{0}; i_sample < num_samples; ++i_sample)
            {
                Col<int> x = x0;
                double t = 0.0;
                while (t < times(it))
                {
                    double r1 = arma::randu();
                    double r2 = arma::randu();

                    Row<double> alpha = model.t_fun(0.0, model.parameters)%model.propensity(x);

                    double a0 = sum(alpha);

                    double tau = -log(r1)/a0;

                    size_t reaction{0};
                    double a1 = alpha(0);

                    while (a1 < r2*a0 && reaction < alpha.n_elem-1)
                    {
                        reaction +=1;
                        a1 += alpha(reaction);
                    }

                    if (t+tau <= times(it))
                    {
                        x = x + model.stoich_mat.col(reaction);
                        t = t + tau;
                    }
                    else
                    {
                        t = times(it);
                    }
                }
                snapshots(it).col(i_sample) = x;
            }
        }

        struct SingleCellData data{ times, snapshots};
        return data;
    };
}
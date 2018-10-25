#pragma once
#include <armadillo>   // Linear Algebra library with similar interface to MATLAB
#include <vector>
#include "Model.h"
#include "cme_util.h"

using namespace arma;

namespace cme{

    class FSPMat
    {
        typedef std::function< Row<double> (double) > TcoefFun ;
        size_t nst,     // number of states
                ns,      // number of species
                nr;      // number of reactions

        std::vector<sp_mat>  term;

        double t = 0;

        TcoefFun tcoeffunc = nullptr;


    public:

        Row<double> tcoef;
        // Functions to get member variables
        int size();
        int nspecies();
        int nreactions();
        sp_mat get_term(size_t i);

        // Constructor
        FSPMat ();
        explicit FSPMat (const struct Model& model);

        // Copy constructor
        FSPMat (FSPMat &Aold);

        // Subroutine to compute A(t)
        FSPMat& operator ()(double t_in);

        // Multiplication with a column vector
        Col<double> operator *(Col<double>& v);
        Col<double> operator *(const Col<double>& v);
    };
}

#include "FSPMat.h"

using namespace arma;

namespace cme {
	int FSPMat::size() {
		return nst;
	}

	int FSPMat::nspecies() {
		return ns;
	}

	int FSPMat::nreactions() {
		return nr;
	}

	sp_mat FSPMat::get_term(size_t i) {
		return term[i];
	}

// Constructor
	FSPMat::FSPMat() {
		nst = 0;
		ns = 0;
		nr = 0;
	}

	FSPMat::FSPMat (const struct Model& model){
		const Mat<int>& Stoich = model.stoich_mat;
		const Row<size_t>& fsp_size = model.FSPSize;

		ns = Stoich.n_rows;
		nr = Stoich.n_cols;
		tcoef = arma::ones(1, nr);

		nst = 1;
		for (size_t i{1}; i <= ns; i++) { nst *= (fsp_size(i - 1) + 1); };

		Row<int> indx = linspace<Row<int> >(0, (int) nst - 1, nst);
		Mat<int> X = ind2sub_fsp(fsp_size, indx);
		Mat<double> propval = model.propensity(X);

		for (size_t mu{0}; mu < nr; mu++) {
			Mat<int> RX = X + repmat(Stoich.col(mu), 1, nst);

			// locations of off-diagonal elements, out-of-bound locations are set to -1
			Row<int> rindx = sub2ind_fsp(fsp_size, RX);

			// off-diagonal elements
			Col<double> prop_keep{propval.col(mu)};
			// eliminate off-diagonal elements that are out-of-bound
			for (size_t i{0}; i < nst; i++) {
				if (rindx(i) == -1) {
					rindx(i) = 0;
					prop_keep(i) = 0.0;
				}
			}

			// prepare locations and values for sparse matrix
			umat locations = conv_to<umat>::from(join_horiz(repmat(indx, 2, 1),
															join_vert(rindx, indx)));

			Col<double> vals{join_vert(-1.0 * propval.col(mu), prop_keep)};

			term.emplace_back(sp_mat(true, locations, vals, nst, nst, true, true));

			tcoeffunc = [&] (double t) {return model.t_fun(t, model.parameters);};

			if (tcoeffunc != nullptr)
			{
				tcoef = tcoeffunc(0.0);
			}
		}
	}

//copy
	FSPMat::FSPMat(FSPMat &Aold) {
		nst = Aold.nst;
		ns = Aold.ns;
		nr = Aold.nr;
		term = Aold.term;
	}

// subroutine to compute A(t)
	FSPMat &FSPMat::operator()(double t_in) {
		t = t_in;
		tcoef = tcoeffunc(t_in);
		return *this;
	}

// define the multiplication of CME matrix with column vector
	Col<double> FSPMat::operator*(Col<double> &v) {
		Col<double> w = zeros(v.size());
		for (size_t i{0}; i < nr; i++) {
			w = w + (tcoef(i)) * (term[i] * v);
		}
		return w;
	}

	Col<double> FSPMat::operator*(const Col<double> &v) {
		Col<double> w = zeros(v.size());
		for (size_t i{0}; i < nr; i++) {
			w = w + (tcoef(i)) * (term[i] * v);
		}
		return w;
	}
}
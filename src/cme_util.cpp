#include "cme_util.h"

namespace cme{
/*
   The following two functions mimic the similar MATLAB functions. Armadillo only supports up to 3 dimensions,
   thus the neccesity of writing our own code.
 */

Mat<int> ind2sub_fsp( const Row<size_t> &nmax, const Row<int> &indx )
{
        uint N = nmax.size();
        uint nst = indx.size();

        Mat<int> X(N, nst);

        int k;
        for (size_t j {1}; j <= nst; j++)
        {
                k = indx(j-1);
                for (size_t i {1}; i <= N; i++)
                {
                        X(i-1, j-1) = k%(nmax(i-1) + 1);
                        k = k/(nmax(i-1) + 1);
                }
        }

        return X;
}

Row<int> sub2ind_fsp( const Row<size_t> &nmax, const Mat<int> &X )
{
        uint N = nmax.size();
        uint nst = X.n_cols;

        Row<int> indx(nst);

        int nprod{1};
        for (size_t j {1}; j <= nst; j++)
        {
                nprod = 1;
                indx(j-1) = 0;
                for ( size_t i {1}; i<= N; i++)
                {
                        if (X(i-1, j-1) < 0 || X(i-1, j-1) > nmax(i-1))
                        {
                                indx(j-1) = -1;
                                break;
                        }
                        indx(j-1) += X(i-1, j-1)*nprod;
                        nprod *= ( nmax(i-1) + 1);
                }
        }

        return indx;
}
}

#include <armadillo>
using namespace arma;
namespace cme{

    /*
The following two functions mimic the similar MATLAB functions. Armadillo only supports up to 3 dimensions,
thus the neccesity of writing our own code.
*/

    Mat<int> ind2sub_fsp( const Row<size_t> &nmax, const Row<int> &indx );

    Row<int> sub2ind_fsp( const Row<size_t> &nmax, const Mat<int> &X );
}


#include "cme_util.h"

namespace cuFSP{

    __device__ __host__
    void indx2state(size_t indx, int *state, size_t dim, size_t *fsp_bounds) {
        for (size_t i{1}; i <= dim; i++) {
            state[i - 1] = indx % (fsp_bounds[i - 1] + 1);
            indx = indx / (fsp_bounds[i - 1] + 1);
        }
    }

    __device__ __host__
    int state2indx(int *state, size_t dim, size_t *fsp_bounds) {
        int indx = 0;
        int nprod = 1;
        for (size_t i{1}; i <= dim; ++i) {
            indx += state[i - 1] * nprod;
            nprod *= (fsp_bounds[i - 1] + 1);
        }
        return indx;
    }
}

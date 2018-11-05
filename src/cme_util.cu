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

    __global__
    void fsp_get_states(int *d_states, size_t dim, size_t n_states, size_t *n_bounds) {

        extern __shared__
        size_t n_bounds_copy[];

        size_t ti = threadIdx.x;
        size_t indx = blockIdx.x * blockDim.x + ti;

        if (ti < dim) {
            n_bounds_copy[ti] = n_bounds[ti];
        }

        __syncthreads();

        if (indx < n_states) {
            indx2state(indx, &d_states[indx * dim], dim, &n_bounds[0]);
        }
    }

    __host__
    __device__
    void reachable_state(int *state, int *rstate, int reaction, int direction,
                         int n_species, int *stoich_val, int *stoich_colidxs, int *stoich_rowptrs) {
        for (int k{0}; k < n_species; ++k) {
            rstate[k] = state[k];
        }
        for (int i = stoich_rowptrs[reaction]; i < stoich_rowptrs[reaction + 1]; ++i) {
            rstate[stoich_colidxs[i]] += direction * stoich_val[i];
        }
    }
}

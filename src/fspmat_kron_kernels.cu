//
// Created by Huy Vo on 11/24/18.
//

#include "fspmat_kron_kernels.h"

namespace cuFSP {

    void SDKronMatSet::destroy() {
        if (vals) {
            cudaFree(vals);
            CUDACHKERR();
        }
        if (offsets) {
            cudaFree(offsets);
            CUDACHKERR();
        }
        if (d_coefs) {
            cudaFree(d_coefs);
            CUDACHKERR();
        }
        if (n_bounds) {
            cudaFree(n_bounds);
            CUDACHKERR();
        }
        num_matrices = 0;
        num_factors = 0;
        mat_data_size = 0;
    }

    void SDKronMatSet::action(double *x, double *y, double *coefs) {
        cudaMemcpy(d_coefs, coefs, num_matrices * sizeof(double), cudaMemcpyHostToDevice);
        CUDACHKERR();

        // Get the max number of threads that can fit to a block
        int max_block_size, num_blocks;
        int device_id;
        cudaGetDevice(&device_id);
        CUDACHKERR();
        cudaDeviceGetAttribute(&max_block_size, cudaDevAttrMaxThreadsPerBlock, device_id);
        CUDACHKERR();
        num_blocks = (int) ceil(n_global / max_block_size * 1.0);

        fspmat_sdkron_mv << < num_blocks, max_block_size, mv_sm_size >> > (num_factors, num_matrices, n_global,
                n_bounds, vals, offsets,
                d_coefs, x, y);
        cudaDeviceSynchronize();
        CUDACHKERR();
    }

    __global__
    void
    fspmat_sdkron_mv(int n_species, int n_reactions, int n_global, int *n_bounds, double *vals, int *offsets,
                     const double *coefs, const double *x,
                     double *y) {
        extern __shared__ double wsp[]; // wsp needs to be large enough for the content of the diagonal of a single sdkmat factor

        int n_left, n_here, n_right, i_left, i_right, i_here, offset, i_val, i_x;
        double y_val, alpha, beta, x_here{0.0};

        int tid = threadIdx.x + blockDim.x * blockIdx.x;

        y_val = 0.0;
        i_val = 0;
        if (tid < n_global)
        {
            x_here = x[tid];
        }
        for (int reaction{0}; reaction < n_reactions; ++reaction) {

            n_left = 1;
            n_right = n_global;

            alpha = 1.0;
            beta = 1.0;
            i_x = tid;

            for (int idim{0}; idim < n_species; ++idim) {
                offset = offsets[reaction * n_species + idim];

                // This block ensures:
                // n_right = n[idim+1]*...*n[n_species-1]
                // n_left = n[0]..n[idim-1]
                n_here = n_bounds[idim] + 1;
                n_right /= n_here;

                // load the content of the idim-th factor
                int num_ph = (int) ceil(n_here / (blockDim.x * 1.0));
                for (int ph = 0; ph < num_ph; ++ph) {
                    if (threadIdx.x + ph * blockDim.x < n_here) {
                        wsp[threadIdx.x + ph * blockDim.x] = vals[i_val + threadIdx.x + ph * blockDim.x];
                    }
                }
                __syncthreads();

                if (tid < n_global) {
                    // Figure out the indices:
                    // i_left = lex(i[0],...,i[idim-1])
                    // i_right = lex(i[idim+1], .., i[n_species])
                    // i_here = i[n_species]
                    i_left = tid % (n_left);
                    i_right = tid / (n_left);
                    i_here = i_right % n_here;
                    i_right = i_right / n_here;


                    if ((i_here + offset >= 0) && (i_here + offset < n_here)) {
                        alpha *= wsp[i_here + offset];
                        i_x += n_left * offset;
                    }

                    if ((i_here + offset < 0) || (i_here + offset >= n_here)) {
                        alpha = 0.0;
                    }

                    beta*=wsp[i_here];
                }
                n_left *= n_here; // n_left = n[0]*..*n[idim]
                i_val += n_here;
            }

            if (tid < n_global) {
                y_val += coefs[reaction]*(x[i_x] * alpha - x_here*beta);
            }
        }
        if (tid < n_global) {
            y[tid] += y_val;
        }
    }

    __host__
    void generate_fsp_mats_sdkron(int n_reactions, int n_species, int *fsp_bounds, CSRMatInt stoich, PropFactorFun pffun,
                                      SDKronMatSet *sdkmatset) {
        sdkmatset->num_matrices = n_reactions;
        sdkmatset->num_factors = n_species;
        sdkmatset->n_global = rect_fsp_num_states(n_species, fsp_bounds);
        sdkmatset->mat_data_size = 0;
        for (int i{0}; i < n_species; ++i) {
            sdkmatset->mat_data_size += (fsp_bounds[i] + 1);
        }
        cudaMalloc(&sdkmatset->n_bounds, sdkmatset->num_factors * sizeof(int));
        CUDACHKERR();
        cudaMemcpy(sdkmatset->n_bounds, fsp_bounds, sdkmatset->num_factors * sizeof(int), cudaMemcpyHostToDevice);
        CUDACHKERR();

        int shared_size = 0;
        for (int i{0}; i < n_species; ++i) {
            shared_size = (shared_size < fsp_bounds[i] + 1) ? fsp_bounds[i] + 1 : shared_size;
        }
        shared_size *= sizeof(double);
        sdkmatset->mv_sm_size = shared_size;

        std::vector<double> h_vals((size_t) sdkmatset->num_matrices * sdkmatset->mat_data_size);
        std::vector<int> h_offsets((size_t) sdkmatset->num_matrices * sdkmatset->num_factors);
        fspmat_sdkron_fill_host_data(n_species, fsp_bounds, n_reactions, stoich, pffun, h_vals.data(),
                                     h_offsets.data());

        cudaMalloc(&sdkmatset->offsets, sdkmatset->num_matrices * sdkmatset->num_factors * sizeof(int));
        CUDACHKERR();
        cudaMalloc(&sdkmatset->vals, sdkmatset->num_matrices * sdkmatset->mat_data_size * sizeof(double));
        CUDACHKERR();
        cudaMemcpy(sdkmatset->offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
        CUDACHKERR();
        cudaMemcpy(sdkmatset->vals, h_vals.data(),sdkmatset->num_matrices * sdkmatset->mat_data_size* sizeof(double), cudaMemcpyHostToDevice);
        CUDACHKERR();

        cudaMalloc(&sdkmatset->d_coefs, n_reactions*sizeof(double));
    }

    __host__
    void
    fspmat_sdkron_fill_host_data(int n_species, int *fsp_bounds, int n_reactions, CSRMatInt stoich, PropFactorFun pffun,
                                 double *vals, int *offsets) {
        int ival = 0;
        for (int reaction = 0; reaction < n_reactions; ++reaction) {
            // Fill the diagonal values
            for (int species{0}; species < n_species; ++species) {
                int n = fsp_bounds[species];
                for (int i{0}; i <= n; ++i) {
                    vals[ival + i] = pffun(i, species, reaction);
                }
                ival += n + 1;
                offsets[n_species * reaction + species] = 0;
            }
            // Fill the offsets
            for (int i = stoich.row_ptrs[reaction]; i < stoich.row_ptrs[reaction + 1]; ++i) {
                offsets[n_species * reaction + stoich.col_idxs[i]] = -1 * stoich.vals[i];
            }
        }
    }
}
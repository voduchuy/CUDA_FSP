//
// Created by Huy Vo on 11/11/18.
//
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include "cme_util.h"
#include "fspmat_csr_kernels.h"
#include <cooperative_groups.h>
#include <cublas.h>

__device__ __host__
double toggle_propensity_factor(int x, int species, int reaction) {
    double prop_val = 1.0;
    switch (reaction) {
        case 0:
            break;
        case 1:
            if (species == 0) {
                prop_val = (double) x;
            }
            break;
        case 2:
            break;
        case 3:
            if (species == 1) {
                prop_val = (double) x;
            }
            break;
        default:
            return 0.0;
    }
    return prop_val;
}

// Generate the Kronecker factors corresponding to reaction k
// Each reaction gives rise to 2 Kronecker-product matrices, one of them is diagonal
// Each sparse factor is represented as a shifted diagonal matrix
// The diagonal and non-diagonal matrices share the same data array
void fsp_component_fill_data(int n_species, const int *fsp_bounds, int reaction, cuFSP::CSRMatInt stoich,
                             cuFSP::SDKronMat *sdkmat) {
    if (reaction >= stoich.n_rows) {
        throw std::runtime_error("fspmat_component_fill_data: requested "
                                 "reaction exceeds the number of rows of stoichiometry matrix.");
    }
    // Fill the diagonal values
    int ival = 0;
    for (int species{0}; species < n_species; ++species) {
        int n = fsp_bounds[species];
        for (int i{0}; i <= n; ++i) {
            sdkmat->vals[ival + i] = toggle_propensity_factor(i, species, reaction);
        }
        ival += n + 1;
        sdkmat->offsets[species] = 0;
    }
    // Fill the offsets
    for (int i = stoich.row_ptrs[reaction]; i < stoich.row_ptrs[reaction + 1]; ++i) {
        sdkmat->offsets[stoich.col_idxs[i]] = -1 * stoich.vals[i];
    }
}

__global__
// Assumption:
// - x and y are arranged in lexicographic order, x(0,0,..), x(1,0, ..), x(2,0,...),...
// - always use square block
// - x and y are of the same size, so each Kronecker factor is a square matrix
// - each Kronecker factor is a shifted diagonal matrix
void kronmat_mv(int dim, int *n_bounds, const cuFSP::SDKronMat sdkmat, const double *x, double *y) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    int n_left, n_here, n_right, i_left, i_right, i_here, offset, i_val;
    double y_val;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    n_left = 1;
    n_right = 1;
    for (int i = 0; i < dim; ++i) {
        n_right *= (n_bounds[i] + 1);
    }

    if (tid < n_right) {
        y[tid] = x[tid];
        g.sync();
        i_val = 0;
        for (int idim{0}; idim < dim; ++idim) {
            offset = sdkmat.offsets[idim];

            // This block ensures:
            // n_right = n[idim+1]*...*n[dim-1]
            // n_left = n[0]..n[idim-1]
            n_here = n_bounds[idim] + 1;
            n_right /= n_here;

            // Figure out the indices:
            // i_left = lex(i[0],...,i[idim-1])
            // i_right = lex(i[idim+1], .., i[dim])
            // i_here = i[dim]
            i_left = tid % (n_left);
            i_right = tid / (n_left);
            i_here = i_right % n_here;
            i_right = i_right / n_here;

            if ((i_here + offset >= 0) && (i_here + offset < n_here)) {
                y_val = sdkmat.vals[i_val + i_here + offset] * y[tid + offset * n_left];
            } else {
                y_val = 0.0;
            }

            n_left *= n_here; // n_left = n[0]*..*n[idim]
            i_val += n_here;
            g.sync();
            y[tid] = y_val;
            g.sync();
        }
    }
}

__global__
// Assumption:
// - x and y are arranged in lexicographic order, x(0,0,..), x(1,0, ..), x(2,0,...),...
// - always use square block
// - x and y are of the same size, so each Kronecker factor is a square matrix
// - each Kronecker factor is a shifted diagonal matrix
void kronmat_mv2(int dim, int *n_bounds, const cuFSP::SDKronMat sdkmat, const double *x, double *y) {
    int n_left, n_here, n_right, i_left, i_right, i_here, offset, i_val, i_x;
    double alpha;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    n_left = 1;
    n_right = 1;

#pragma unroll
    for (int i = 0; i < dim; ++i) {
        n_right *= (n_bounds[i] + 1);
    }

    if (tid < n_right) {
        alpha = 1.0;
        i_x = tid;
        i_val = 0;

#pragma unroll
        for (int idim{0}; idim < dim; ++idim) {
            offset = sdkmat.offsets[idim];

            // This block ensures:
            // n_right = n[idim+1]*...*n[dim-1]
            // n_left = n[0]..n[idim-1]
            n_here = n_bounds[idim] + 1;
            n_right /= n_here;

            // Figure out the indices:
            // i_left = lex(i[0],...,i[idim-1])
            // i_right = lex(i[idim+1], .., i[dim])
            // i_here = i[dim]
            i_left = tid % (n_left);
            i_right = tid / (n_left);
            i_here = i_right % n_here;
            i_right = i_right / n_here;

            if ((i_here + offset >= 0) && (i_here + offset < n_here)) {
                alpha *= sdkmat.vals[i_val + i_here + offset];
                i_x += n_left * offset;
            }

            if ((i_here + offset < 0) || (i_here + offset >= n_here)) {
                alpha = 0.0;
            }

            n_left *= n_here; // n_left = n[0]*..*n[idim]
            i_val += n_here;
        }
        y[tid] += x[i_x] * alpha;
    }
}

__global__
// Assumption:
// - x and y are arranged in lexicographic order, x(0,0,..), x(1,0, ..), x(2,0,...),...
// - always use square block
// - x and y are of the same size, so each Kronecker factor is a square matrix
// - each Kronecker factor is a shifted diagonal matrix
void kronmat_mv3(int dim, int n_global, int *n_bounds, const cuFSP::SDKronMat sdkmat, const double *x, double *y) {
    extern __shared__ double wsp[]; // wsp needs to be large enough for the content of the diagonal of a single sdkmat factor

    int n_left, n_here, n_right, i_left, i_right, i_here, offset, i_val, i_x;
    double y_val, alpha;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    n_left = 1;
    n_right = n_global;

    alpha = 1.0;
    i_x = tid;
    i_val = 0;
    for (int idim{0}; idim < dim; ++idim) {
        offset = sdkmat.offsets[idim];

        // This block ensures:
        // n_right = n[idim+1]*...*n[dim-1]
        // n_left = n[0]..n[idim-1]
        n_here = n_bounds[idim] + 1;
        n_right /= n_here;

        // load the content of the idim-th factor
        int num_ph = (int) ceil(n_here / (blockDim.x * 1.0));
        for (int ph = 0; ph < num_ph; ++ph) {
            if (threadIdx.x + ph * blockDim.x < n_here) {
                wsp[threadIdx.x + ph * blockDim.x] = sdkmat.vals[i_val + threadIdx.x + ph * blockDim.x];
            }
        }
        __syncthreads();

        if (tid < n_global) {
            // Figure out the indices:
            // i_left = lex(i[0],...,i[idim-1])
            // i_right = lex(i[idim+1], .., i[dim])
            // i_here = i[dim]
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
        }
        n_left *= n_here; // n_left = n[0]*..*n[idim]
        i_val += n_here;
    }

    if (tid < n_global) {
        y[tid] += x[i_x] * alpha;
    }
}

int main() {
    int n_species = 2, n_reactions = 4;

    int *fsp_bounds;
    cudaMallocManaged(&fsp_bounds, 2 * sizeof(int));
    CUDACHKERR();
    fsp_bounds[0] = 1 << 11 - 1;
    fsp_bounds[1] = 1 << 11 - 1;
    int n_states = cuFSP::rect_fsp_num_states(n_species, fsp_bounds);

    int stoich_vals[] = {1, -1, 1, -1};
    int stoich_colidxs[] = {0, 0, 1, 1};
    int stoich_rowptrs[] = {0, 1, 2, 3, 4};
    cuFSP::CSRMatInt stoich;
    stoich.vals = &stoich_vals[0];
    stoich.col_idxs = &stoich_colidxs[0];
    stoich.row_ptrs = &stoich_rowptrs[0];
    stoich.n_rows = 4;
    stoich.n_cols = 2;
    stoich.nnz = 4;

    int kron_data_size{0};
    for (int i{0}; i < n_species; ++i) {
        kron_data_size += (fsp_bounds[i] + 1);
    }

    cuFSP::SDKronMat sdkmat;
    sdkmat.d = n_species;
    cudaMallocManaged(&sdkmat.offsets, n_species * sizeof(int));
    CUDACHKERR();
    cudaMallocManaged(&sdkmat.vals, kron_data_size * sizeof(double));
    CUDACHKERR();
    fsp_component_fill_data(n_species, fsp_bounds, 1, stoich, &sdkmat);
    CUDACHKERR();

    double *d_x, *d_y1, *d_y2, *d_y3;
    cudaMallocManaged(&d_x, n_states * sizeof(double));
    CUDACHKERR();
    cudaMallocManaged(&d_y1, n_states * sizeof(double));
    CUDACHKERR();
    cudaMallocManaged(&d_y2, n_states * sizeof(double));
    CUDACHKERR();
    cudaMallocManaged(&d_y3, n_states * sizeof(double));
    CUDACHKERR();
    for (int i{0}; i < n_states; ++i) {
        d_x[i] = 1.0;
        d_y1[i] = 0.0;
        d_y2[i] = 0.0;
        d_y3[i] = 0.0;
    }

    int numBlocks = (int) std::ceil(n_states / 1024.0);
////    kronmat_mv<<<numBlocks, 1024>>>(n_species,fsp_bounds, sdkmat, d_x, d_y1);
//    void* kmvargs[] = {
//            (void*) &n_species,
//            (void*) fsp_bounds,
//            (void*) &sdkmat,
//            (void*) d_x,
//            (void*) d_y1
//    };
//    cudaLaunchCooperativeKernel(&kronmat_mv, numBlocks, 1024, kmvargs);
//    cudaDeviceSynchronize();
//    CUDACHKERR();
//
//    double x = 1.0;
//    for (int i = 0; i < n_states; ++i){
//        if (std::abs(d_y1[i] - x) > 1.0e-14){
//            std::cout << "y1[i] inaccurate at i = " << i << " with value " << d_y1[i] << " and true value " << x <<"\n";
//        }
//        x += 1.0;
//        if (x > 128.0){
//            x = 0.0;
//        }
//    }

    kronmat_mv2 << < numBlocks, 1024 >> > (n_species, fsp_bounds, sdkmat, d_x, d_y2);
    cudaDeviceSynchronize();
    CUDACHKERR();

    {
        double x = 1.0;
        for (int i = 0; i < n_states; ++i) {
            if (std::abs(d_y2[i] - x) > 1.0e-14) {
                std::cout << "y2[i] inaccurate at i = " << i << " with value " << d_y2[i] << " and true value " << x
                          << "\n";
            }
            x += 1.0;
            if (x > fsp_bounds[0]) {
                x = 0.0;
            }
        }
    }

    int shared_size = 0;
    for (int i{0}; i < n_species; ++i) {
        shared_size = (shared_size < fsp_bounds[i] + 1) ? fsp_bounds[i] + 1 : shared_size;
    }
    shared_size *= sizeof(double);
    kronmat_mv3 << < numBlocks, 1024, shared_size >> > (n_species, n_states, fsp_bounds, sdkmat, d_x, d_y3);
    cudaDeviceSynchronize();
    CUDACHKERR();

    {
        double x = 1.0;
        for (int i = 0; i < n_states; ++i) {
            if (std::abs(d_y3[i] - x) > 1.0e-14) {
                std::cout << "y3[i] inaccurate at i = " << i << " with value " << d_y3[i] << " and true value " << x
                          << "\n";
            }
            x += 1.0;
            if (x > fsp_bounds[0]) {
                x = 0.0;
            }
        }
    }

    cublasDaxpy(n_states, -1.0, d_y3, 1, d_y2, 1);
    CUDACHKERR();
    double error_l2 = cublasDnrm2(n_states, d_y2, 1);
    CUDACHKERR();
    std::cout << "Difference between kernels 3 & 2: " << error_l2 << "\n";

//    cublasDaxpy(n_states, -1.0, d_y1, 1, d_y3, 1); CUDACHKERR();
//    error_l2 = cublasDnrm2(n_states, d_y3, 1); CUDACHKERR();
//    std::cout << "Difference between kernels 1 & 3: " << error_l2 << "\n";
//
//

    cudaFree(sdkmat.offsets);
    cudaFree(sdkmat.vals);
    cudaFree(d_x);
    CUDACHKERR();
    cudaFree(d_y1);
    CUDACHKERR();
    cudaFree(d_y2);
    CUDACHKERR();
    cudaFree(d_y3);
    CUDACHKERR();
    return 0;
}
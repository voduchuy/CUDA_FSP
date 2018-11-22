//
// Created by Huy Vo on 11/11/18.
//
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include "cme_util.h"
#include "fspmat_csr_kernels.h"

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
    }
    return prop_val;
}

// Generate the Kronecker factors corresponding to reaction k
// Each reaction gives rise to 2 Kronecker-product matrices
// Each sparse factor is represented in ELL format
void fsp_component_kronmat_generate(int n_species, const int *fsp_bounds, int reaction, cuFSP::CSRMatInt stoich,
                                    double *val, int *colidx, int *kf_ptr) {
    int val_offset = 0;
    // Fill the diagonal matrix
    for (int species{0}; species < n_species; ++species) {
        kf_ptr[species] = val_offset;
        int n = fsp_bounds[species];

        for (int i{0}; i <= n; ++i) {
            val[val_offset + i] = -1.0 * toggle_propensity_factor(i, species, reaction);
            colidx[val_offset + i] = i;
        }
        val_offset += n + 1;
    }

    // Fill the off-diagonal matrix
    std::vector<int> changed_species((size_t) n_species, 0);
    std::vector<int> jump((size_t) n_species, 0);

    for (int i{stoich.row_ptrs[reaction]}; i < stoich.row_ptrs[reaction + 1]; ++i) {
        changed_species.at((size_t) stoich.col_idxs[i]) = 1;
        jump.at((size_t) stoich.col_idxs[i]) = stoich.vals[i];
    }

    int val_offset_old = val_offset;
    for (int species{0}; species < n_species; ++species) {
        kf_ptr[n_species + species] = val_offset - val_offset_old;
        int n = fsp_bounds[species];

        if (changed_species[(size_t) species] == 0) {
            for (int i{0}; i <= n; ++i) {
                val[val_offset + i] = toggle_propensity_factor(i, species, reaction);
                colidx[val_offset + i] = i;
            }
        } else {
            for (int i{0}; i <= n; ++i) {
                if ((i - jump[species] >= 0) && (i - jump[species] <= n)) {
                    val[val_offset + i] = toggle_propensity_factor(i - jump[species], species, reaction);
                    colidx[val_offset + i] = i - jump[species];
                } else {
                    val[val_offset + i] = 0.0;
                    colidx[val_offset + i] = 0;
                }
            }
        }
        val_offset += n + 1;
    }
}

__global__
// Assumption:
// - x and y are arranged in lexicographic order, x(0,0,..), x(1,0, ..), x(2,0,...),...
// - always use square block
// - x and y are of the same size, so each Kronecker factor is a square matrix
// - each Kronecker factor is a sparse matrix with only 1 nonzero per row
void kronmat_mv(int dim, int *n_bounds, double *val, int *colidx, int *kf_ptrs, double *x, double *y) {
    extern __shared__ double wsp[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int tix = threadIdx.x;
    int tiy = threadIdx.y;
    int tile_width = blockDim.x;

    int n_rows, n_cols, n_left;
    double *x_tile, *Aval;
    int *Acolidx;

    n_cols = 1;
    for (int i{0}; i < dim; ++i) {
        n_cols *= (n_bounds[i] + 1);
    }
    //---------------------------------------------------------------------------
    // Contract the 0-th dimension with the 0-th factor
    //---------------------------------------------------------------------------
    // Using multiplication of the kron factor with mode-0 unfolding of x
    n_rows = n_bounds[0] + 1; // number of rows in the unfolding
    n_cols = n_cols/n_rows; // number of cols in the unfolding


    x_tile = wsp;
    Aval = &x_tile[tile_width*tile_width];
    Acolidx = (int *) &Aval[n_rows];

    // Collaborative loading of data from kronecker factor
    for (int lane{0}; lane <= n_rows/tile_width; ++lane){
        if (tid + tile_width*lane < n_rows) {
            Aval[tid + tile_width*lane] = val[kf_ptrs[0] + tid + tile_width*lane];
        }
        __syncthreads();
        if (tid + tile_width*lane < n_rows) {
            Acolidx[tid + tile_width*lane] = colidx[kf_ptrs[0] + tid + tile_width*lane];
        }
        __syncthreads();
    }

    for (int ph = 0; ph <= n_rows / tile_width; ++ph) {
        // Collaborative loading of data from x into shared memory
        int i_row_x = ph * tile_width + tix;
        int i_col_x = blockIdx.x * blockDim.y + tiy;
        if (i_row_x < n_rows && i_col_x < n_cols) {
            // Find the corresponding index in x
            x_tile[tix + tile_width * tiy] = x[i_row_x + n_rows * i_col_x];
        } else {
            x_tile[tix + tile_width * tiy] = 0.0;
        }

        __syncthreads();
        // Update the entries of y
        if (i_col_x < n_cols) {
            for (int ph1{0}; ph1 <= n_rows / tile_width; ++ph1) {
                // Figure out which entry in shared memory is needed
                int i_row_y = ph1 * tile_width + tix;
                if (i_row_y < n_rows) {
                    int i_row_x_tile = Acolidx[i_row_y];
                    if (i_row_x_tile / tile_width == ph) {
                        i_row_x_tile = i_row_x_tile % tile_width;
                        int i_col_y = i_col_x;
                        y[i_row_y + n_rows * i_col_y] = Aval[i_row_y] * x_tile[i_row_x_tile + tile_width * tiy];
                    }
                }
            }
        }
    }
    // Assign n_cols to the total number of elements in x
    n_cols = n_cols*n_rows;
    n_left = n_rows;
    //---------------------------------------------------------------------------
    // Contract the i-th dimension with the i-th factor,i > 0
    //---------------------------------------------------------------------------
    // Loop through dimensions
    for (int idim{1}; idim < dim; ++idim){
        n_rows = n_bounds[idim] + 1;
        n_cols = n_cols/n_rows;

        // Re-adjust pointer to the new Kronecker factor
        Acolidx = (int *) &Aval[n_rows];

        // Collaborative loading of data from Kronecker factor
        for (int lane{0}; lane <= n_rows/tile_width; ++lane){
            if (tid + tile_width*lane < n_rows) {
                Aval[tid + tile_width*lane] = val[kf_ptrs[idim] + tid + tile_width*lane];
            }
            __syncthreads();
            if (tid + tile_width*lane < n_rows) {
                Acolidx[tid + tile_width*lane] = colidx[kf_ptrs[idim] + tid + tile_width*lane];
            }
            __syncthreads();
        }

        for (int ph{0}; ph <= n_rows/tile_width; ++ph){
            // Collaborative loading of data in y to shared memory
            int i_row_x = ph * tile_width + tiy;
            int i_col_x = blockIdx.x * blockDim.x + tix;
            if (i_row_x < n_rows && i_col_x < n_cols){
                // Seperate i_col_y into left and right indices, i.e.
                // converting ind(i0,..,i_{k-1}, i_{k+1},..,i_{d-1}) to ind(i0,..,i_{k-1}) and ind(i_{k+1},..,i_{d-1})
                int i_left = i_col_x % n_left;
                int i_right = i_col_x / n_left;
                // copy the current value of y to shared memory
                x_tile[tiy + tile_width*tix] = y[i_left + n_left*i_row_x + n_left*n_rows*i_right];
            }else{
                x_tile[tiy + tile_width*tix] = 0.0;
            }

            __syncthreads();
            // Update the entries of y
            if (i_col_x < n_cols) {
                for (int ph1{0}; ph1 <= n_rows / tile_width; ++ph1) {
                    // Figure out which entry in shared memory is needed
                    int i_row_y = ph1 * tile_width + tiy;
                    if (i_row_y < n_rows) {
                        int i_row_x_tile = Acolidx[i_row_y];
                        if (i_row_x_tile / tile_width == ph) {
                            i_row_x_tile = i_row_x_tile % tile_width;
                            int i_col_y = i_col_x;
                            y[i_row_y + n_rows * i_col_y] = Aval[i_row_y] * x_tile[i_row_x_tile + tile_width * tix];
                        }
                    }
                }
            }
        }
        n_left *= n_rows;
        n_cols = n_cols*n_rows;
    }
}

int main() {
    int n_species = 2, n_reactions = 4;

    int *fsp_bounds;
    cudaMallocManaged(&fsp_bounds, 2 * sizeof(int));
    CUDACHKERR();
    fsp_bounds[0] = 1 << 10;
    fsp_bounds[1] = 1 << 11;
    int n_states = (fsp_bounds[0] + 1) * (fsp_bounds[1] + 1);

    int stoich_vals[] = {1, -1, 1, -1};
    int stoich_colidxs[] = {0, 0, 1, 1};
    int stoich_rowptrs[] = {0, 1, 2, 3, 4};
    cuFSP::CSRMatInt stoich;
    stoich.vals = &stoich_vals[0];
    stoich.col_idxs = &stoich_colidxs[0];
    stoich.row_ptrs = &stoich_rowptrs[0];
    stoich.n_rows = 4;
    stoich.n_cols = 2;

    int kron_data_size{0};
    for (int i{0}; i < n_species; ++i) {
        kron_data_size += (fsp_bounds[i] + 1);
    }

    std::vector<double> val((size_t) 2 * kron_data_size, 0.0);
    std::vector<int> colidx((size_t) 2 * kron_data_size, 0);
    std::vector<int> kf_ptrs((size_t) 2 * n_species, 0);

    fsp_component_kronmat_generate(n_species, &fsp_bounds[0], 3, stoich, val.data(), colidx.data(), kf_ptrs.data());
//    for (int i{0}; i < val.size(); ++i){
//        std::cout << val[i] << " " << colidx[i] << " \n";
//    }

    double *d_val, *d_x, *d_y;
    int *d_colidx, *d_kf_ptrs;
    cudaMalloc(&d_val, val.size() * sizeof(double));
    CUDACHKERR();
    cudaMalloc(&d_colidx, colidx.size() * sizeof(int));
    CUDACHKERR();
    cudaMalloc(&d_kf_ptrs, kf_ptrs.size() * sizeof(int));
    CUDACHKERR();
    cudaMallocManaged(&d_x, n_states * sizeof(double));
    CUDACHKERR();
    cudaMallocManaged(&d_y, n_states * sizeof(double));
    CUDACHKERR();

    for (int i{0}; i < n_states; ++i) {
        d_x[i] = 1.0;
        d_y[i] = 0.0;
    }

    cudaMemcpy(d_val, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice);
    CUDACHKERR();
    cudaMemcpy(d_colidx, colidx.data(), colidx.size() * sizeof(int), cudaMemcpyHostToDevice);
    CUDACHKERR();
    cudaMemcpy(d_kf_ptrs, kf_ptrs.data(), kf_ptrs.size() * sizeof(int), cudaMemcpyHostToDevice);
    CUDACHKERR();

    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 32;
    int numBlocks = (int) std::ceil((fsp_bounds[1] + 1) / 32.0);
    int shared_mem_size = 1024 * sizeof(double) + (fsp_bounds[1] + 1) * (sizeof(double) + sizeof(int));
    std::cout << "Shared mem size = " << shared_mem_size << "\n";
    kronmat_mv << < numBlocks, blockDim, shared_mem_size >> >
                                         (n_species, &fsp_bounds[0], d_val + kron_data_size, d_colidx + kron_data_size,
                                                 d_kf_ptrs + n_species, d_x, d_y);
    cudaDeviceSynchronize();
    CUDACHKERR();

//    for (int i{0}; i < n_states; ++i) {
//        std::cout << d_y[i] << " \n";
//    }

    cudaFree(d_val);
    CUDACHKERR();
    cudaFree(d_colidx);
    CUDACHKERR();
    cudaFree(d_kf_ptrs);
    CUDACHKERR();
    cudaFree(d_x);
    CUDACHKERR();
    cudaFree(d_y);
    CUDACHKERR();
    return 0;
}
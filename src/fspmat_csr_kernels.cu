//
// Created by huy on 11/3/18.
//
#include "fspmat_csr_kernels.h"
namespace cuFSP{

    __global__
    void fspmat_csr_get_nnz_per_row(int *nnz_per_row, int *off_indx, int *states, int reaction, int n_rows,
                                    int n_species, int *fsp_bounds,
                                    int *stoich_vals, int *stoich_colidxs, int *stoich_rowptrs) {
        extern __shared__ int wsp[];

        int tix = threadIdx.x;
        int tid = blockDim.x * blockIdx.x + tix;

        int *fsp_bounds_copy = &wsp[0];

        if (tix < n_species) {
            fsp_bounds_copy[tix] = fsp_bounds[tix];
        }

        __syncthreads();

        int *state;

        if (tid < n_rows) {

            state = &states[tid * n_species];

//            indx2state(tid, &state[0], n_species, fsp_bounds_copy);

            reachable_state(state, state, reaction, -1,
                             n_species, stoich_vals, stoich_colidxs, stoich_rowptrs);

            bool reachable = true;
            for (size_t k{0}; k < n_species; ++k) {
                reachable = reachable && ((state[k] >= 0) && (state[k] <= fsp_bounds_copy[k]));
            }

            nnz_per_row[tid] = 1;
            if (reachable) {
                off_indx[tid] = state2indx(state, n_species, fsp_bounds_copy);
                nnz_per_row[tid] += 1;
            } else {
                off_indx[tid] = -1;
            }

            reachable_state(state, state, reaction, 1,
                            n_species, stoich_vals, stoich_colidxs, stoich_rowptrs);
        }
    }

    __global__

    void
    fspmat_csr_fill_data(double *values, int *col_indices, int *row_ptrs, int n_rows, int reaction,
                         int *off_diag_indices, int *states, int dim, PropFun propensity) {

        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        int off_diag_indx, rowptr, i_diag, i_offdiag;
        int *state;

        if (tid < n_rows) {
            off_diag_indx = off_diag_indices[tid];
            rowptr = row_ptrs[tid];

            if (off_diag_indx >= 0) {

                if (off_diag_indx > tid) {
                    i_diag = rowptr;
                    i_offdiag = rowptr + 1;
                } else {
                    i_diag = rowptr + 1;
                    i_offdiag = rowptr;
                }

                state = states + dim * tid;
                values[i_diag] = propensity(state, reaction);
                values[i_diag] *= -1.0;

                state = states + dim* off_diag_indx;
                values[i_offdiag] = propensity(state, reaction);

                col_indices[i_diag] = tid;
                col_indices[i_offdiag] = off_diag_indx;
            } else {
                state = states + dim * tid;
                values[rowptr] = propensity(state, reaction);
                values[rowptr] *= -1.0;
                col_indices[rowptr] = tid;
            }
        }
    }

    void CUDACSRMatSet::destroy() {
        for (int i{0}; i < term.size(); ++i) {
            if (term[i].col_idxs) {
                cudaFree(term[i].col_idxs);
                CUDACHKERR();
            }
            if (term[i].row_ptrs) {
                cudaFree(term[i].row_ptrs);
                CUDACHKERR();
            }
            if (term[i].vals) {
                cudaFree(term[i].vals);
                CUDACHKERR();
            }
            term[i].col_idxs = nullptr;
            term[i].row_ptrs = nullptr;
            term[i].vals = nullptr;
        }
        if (descr) {
            cusparseDestroyMatDescr(descr);
            CUDACHKERR();
        }
        if (handle) {
            cusparseDestroy(handle);
        }
    }

    void CUDACSRMatSet::action(double *x, double *y, double *coefs) {
        cusparseStatus_t stat;
        for (int i{0}; i < term.size(); ++i) {
            stat = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, term[i].n_rows, term[i].n_rows, term[i].nnz,
                                  &coefs[i],
                                  descr,
                                  term[i].vals, term[i].row_ptrs, term[i].col_idxs,
                                  x,
                                  &h_one,
                                  y);
            assert (stat == CUSPARSE_STATUS_SUCCESS);
            CUDACHKERR();
        }
    }

    void generate_fsp_mats_cuda_csr(int *states, int n_states, int n_reactions, int n_species, int *fsp_dim,
                                    CSRMatInt stoich, PropFun prop_func, CUDACSRMatSet *csr) {
        cusparseCreate(&csr->handle);
        CUDACHKERR();
        cusparseCreateMatDescr(&csr->descr);
        CUDACHKERR();
        cusparseSetMatType(csr->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        CUDACHKERR();
        cusparseSetMatIndexBase(csr->descr, CUSPARSE_INDEX_BASE_ZERO);
        CUDACHKERR();

        int *d_stoich_vals, *d_stoich_colidxs, *d_stoich_rowptrs;
        cudaMalloc(&d_stoich_vals, stoich.row_ptrs[stoich.n_rows] * sizeof(int));
        cudaMalloc(&d_stoich_colidxs, stoich.row_ptrs[stoich.n_rows] * sizeof(int));
        cudaMalloc(&d_stoich_rowptrs, (stoich.n_rows + 1) * sizeof(int));

        cudaMemcpy(d_stoich_vals, stoich.vals, stoich.row_ptrs[stoich.n_rows] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_stoich_colidxs, stoich.col_idxs, stoich.row_ptrs[stoich.n_rows] * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_stoich_rowptrs, stoich.row_ptrs, (stoich.n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Temporary workspace for matrix generation
        int *iwsp;
        cudaMallocManaged(&iwsp, n_states * sizeof(int));
        CUDACHKERR();

        // Get the max number of threads that can fit to a block
        int max_block_size, num_blocks;
        int device_id;

        cudaGetDevice(&device_id);
        CUDACHKERR();
        cudaDeviceGetAttribute(&max_block_size, cudaDevAttrMaxThreadsPerBlock, device_id);
        CUDACHKERR();

        num_blocks = (int) std::ceil(n_states / (max_block_size * 1.0));

        csr->term.resize((size_t) n_reactions);

        // Generate the state space
        fsp_get_states << < num_blocks, max_block_size, n_species * sizeof(int) >> >
                                                        (states, n_species, n_states, fsp_dim);
        CUDACHKERR();
        cudaDeviceSynchronize();

        for (int ir{0}; ir < n_reactions; ++ir) {
            csr->term.at(ir).n_cols = n_states;
            csr->term.at(ir).n_rows = n_states;

            // Initialize CSR data structure for the ir-th matrix
            cudaMalloc((void **) &((csr->term.at(ir)).row_ptrs), (n_states + 1) * sizeof(int));
            CUDACHKERR();

            // Count nonzero entries and store off-diagonal col indices to the temporary workspace
            fspmat_csr_get_nnz_per_row << < num_blocks, max_block_size, n_species * sizeof(int) >> > (csr->term[ir].row_ptrs +
                                                                                                      1, iwsp, states, ir, n_states, n_species, fsp_dim,
                    d_stoich_vals, d_stoich_colidxs, d_stoich_rowptrs);
            CUDACHKERR();
            cudaDeviceSynchronize();
            CUDACHKERR();

            // Use cumulative sum to determine the values of the row pointers in CSR
            int h_zero = 0;
            cudaMemcpy(csr->term[ir].row_ptrs, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
            CUDACHKERR();
            thrust::device_ptr<int> ptr(csr->term[ir].row_ptrs);
            thrust::inclusive_scan(ptr, ptr + (n_states + 1), ptr);
            CUDACHKERR();
            cudaDeviceSynchronize();
            CUDACHKERR();

            // Fill the column indices and values
            int nnz;
            cudaMemcpy(&nnz, csr->term[ir].row_ptrs + n_states, sizeof(int), cudaMemcpyDeviceToHost);
            CUDACHKERR();
            csr->term[ir].nnz = nnz;

            cudaMalloc((void **) &(csr->term[ir].vals), nnz * sizeof(double));
            CUDACHKERR();
            cudaMalloc((void **) &(csr->term[ir].col_idxs), nnz * sizeof(int));
            CUDACHKERR();

            fspmat_csr_fill_data << < num_blocks, max_block_size >> >
                                                  (csr->term[ir].vals, csr->term[ir].col_idxs, csr->term[ir].row_ptrs, n_states, ir, iwsp, states, n_species,
                                                          prop_func);
            CUDACHKERR();


            cudaDeviceSynchronize();
            CUDACHKERR();
        }

        cudaFree(d_stoich_colidxs);
        CUDACHKERR();
        cudaFree(d_stoich_rowptrs);
        CUDACHKERR();
        cudaFree(d_stoich_vals);
        CUDACHKERR();
        cudaFree(iwsp);
        CUDACHKERR();
    }
}
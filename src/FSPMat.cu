#include "FSPMat.h"

using namespace arma;

namespace cuFSP {

    size_t FSPMat::get_n_rows() {
        return nst;
    }

    size_t FSPMat::get_n_species() {
        return ns;
    }

    size_t FSPMat::get_n_reactions() {
        return nr;
    }

    cuda_csr_mat* FSPMat::get_term(size_t i) {
        return &term[i];
    }

// Constructor
    // Precondition:
    // stoich stores the stoichiometry matrix, assumed to be in CSR format, with each row for each reaction
    FSPMat::FSPMat(int *states, size_t n_states, size_t n_reactions, size_t n_species, size_t *fsp_dim,
                             cuda_csr_mat_int stoich, TcoefFun t_func, PropFun prop_func) {

        // Initialize cuSparse handle and bind to stream
        cusparseCreate(&cusparse_handle);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cusparseSetStream(cusparse_handle, stream);

        // Initialize cuSparse descriptor
        cusparseCreateMatDescr(&cusparse_descr);
        cusparseSetMatIndexBase(cusparse_descr,CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL );

        // Temporary workspace for matrix generation
        int *iwsp;
        cudaMalloc(&iwsp, n_states * sizeof(int));

        // Get the max number of threads that can fit to a block
        int max_block_size, num_blocks;

        int device_id;

        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&max_block_size, cudaDevAttrMaxThreadsPerBlock, device_id);

        num_blocks = (size_t) std::ceil(n_states / (max_block_size * 1.0));

        // Initialize dimensions
        nst = n_states;
        nr = n_reactions;
        ns = n_species;

        term.resize(nr);

        // Generate the state space
        fsp_get_states << < num_blocks, max_block_size, ns * sizeof(size_t) >> > (states, ns, nst, fsp_dim);

        for (size_t ir{0}; ir < nr; ++ir) {
            term[ir].n_cols = nst;
            term[ir].n_rows = nst;
            // Initialize CSR data structure for the ir-th matrix
            cudaMalloc(&term[ir].vals, n_states * sizeof(double));
            cudaMalloc(&term[ir].col_idxs, n_states * sizeof(int));
            cudaMalloc(&term[ir].row_ptrs, n_states * sizeof(int));

            // Count nonzero entries and store off-diagonal col indices to the temporary workspace
            term[ir].row_ptrs = 0;
            size_t shared_size = ns*sizeof(size_t) +                                    // mem for fsp_dim
                                 ns*sizeof(int) + stoich.row_ptrs[nr]*2*sizeof(int);    // mem for stoich
            fspmat_component_get_nnz_per_row << < num_blocks, max_block_size, shared_size>> >
                                                                              (term[ir].row_ptrs +
                                                                               1, iwsp, (int) ir, nst, ns, states, fsp_dim, stoich);
            // Use cumulative sum to determine the values of the row pointers in CSR
            thrust::inclusive_scan(term[ir].row_ptrs, term[ir].row_ptrs + (nst + 1), term[ir].row_ptrs);
            // Fill out the column indices and values
            fspmat_component_fill_data_csr << < num_blocks, max_block_size >> >
                                                         (term[ir].vals, term[ir].col_idxs, term[ir].row_ptrs, nst, ir, iwsp, states, ns,
                                                                 prop_func);
        }

        cudaFree(iwsp);
    }

    void action(double t, double *x, double *y){

    }

    // Destructor
    FSPMat::~FSPMat() {
        for (size_t i{0}; i < nr; ++i) {
            cudaFree(term.at(i).col_idxs);
            cudaFree(term.at(i).row_ptrs);
            cudaFree(term.at(i).vals);
        }
        cusparseDestroy(cusparse_handle);
        cudaStreamDestroy(stream);
        cusparseDestroyMatDescr(cusparse_descr);
    }

    __global__
    void fsp_get_states(int *d_states, size_t dim, size_t n_states, size_t *n_bounds) {
        extern __shared__ size_t n_bounds_copy[];

        size_t ti = threadIdx.x;
        size_t indx = blockIdx.x * blockDim.x + ti;

        if (ti < dim)
            n_bounds_copy[ti] = n_bounds[ti];

        __syncthreads();

        if (indx < n_states) {
            indx2state(indx, &d_states[indx * dim], dim, &n_bounds_copy[0]);
        }
    }

    __host__ __device__
    void reachable_state(int *state, int *rstate, int reaction, int direction,
                         int n_species, int *stoich_val, int *stoich_colidxs, int *stoich_rowptrs) {
        for (int k{0}; k < n_species; ++k) {
            rstate[k] = state[k];
        }
        for (int i = stoich_rowptrs[reaction]; i < stoich_rowptrs[reaction] + 1; ++i) {
            rstate[stoich_colidxs[i]] += direction * stoich_val[i];
        }
    }

    __global__
    void
    fspmat_component_get_nnz_per_row(int *nnz_per_row, int *off_indx, int reaction, size_t n_rows,
                                     size_t n_species, int *states, size_t *fsp_bounds, cuda_csr_mat_int stoich) {
        size_t tix = threadIdx.x;
        size_t tid = blockDim.x * blockIdx.x + tix;

        extern __shared__ size_t wsp[];

        size_t *fsp_bounds_copy = &wsp[0];

        size_t stoich_n_rows = stoich.n_rows;
        int stoich_nnz = stoich.row_ptrs[stoich_n_rows];

        int *stoich_vals = (int *) fsp_bounds_copy[n_species];
        int *stoich_colidxs = (int *) stoich_vals[stoich_nnz];
        int *stoich_rowptrs = (int *) stoich_colidxs[stoich_nnz];

        if (tix == 0) {
            for (int k{0}; k < stoich_n_rows + 1; ++k) {
                stoich_rowptrs[k] = stoich.row_ptrs[k];
            }

            for (int k{0}; k < stoich_nnz; ++k) {
                stoich_vals[k] = stoich.vals[k];
                stoich_colidxs[k] = stoich.col_idxs[k];
            }

            for (size_t k{0}; k < n_species; ++k) {
                fsp_bounds_copy[k] = fsp_bounds[k];
            }
        }

        __syncthreads();


        int *state;


        if (tid < n_rows) {
            state = &states[tid * n_species];

            indx2state(tid, &state[0], n_species, fsp_bounds_copy);
            reachable_state(state, state, reaction, -1,
                            n_species, stoich_vals, stoich_colidxs, stoich_rowptrs);

            bool reachable = true;
            for (size_t k{0}; k < n_species; ++k) {
                reachable = reachable && ((state[k] >= 0) || (state[k] <= fsp_bounds_copy[k]));
            }

            nnz_per_row[tid] = 1;
            if (reachable) {
                state2indx(state, off_indx[tid], n_species, fsp_bounds_copy);
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
    fspmat_component_fill_data_csr(double *values, int *col_indices, int *row_ptrs, size_t n_rows, int reaction,
                                   int *off_diag_indices, int *states, size_t dim, PropFun propensity) {
        size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

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
                col_indices[i_diag] = (int) tid;
                values[i_diag] *= -1.0;

                state = states + dim * off_diag_indx;
                values[i_offdiag] = propensity(state, reaction);
                col_indices[i_offdiag] = off_diag_indx;
            } else {
                state = states + dim * tid;
                values[rowptr] = propensity(state, reaction);
                values[rowptr] *= -1.0;
                col_indices[rowptr] = (int) tid;
            }

        }
    }
}
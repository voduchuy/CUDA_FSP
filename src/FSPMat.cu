#include "FSPMat.h"
#include "../../../../../../usr/local/cuda/include/driver_types.h"

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

    cuda_csr_mat *FSPMat::get_term(size_t i) {
        return &term[i];
    }

// Constructor
    // Precondition:
    // stoich stores the stoichiometry matrix, assumed to be in CSR format, with each row for each reaction
    FSPMat::FSPMat
//    (cusparseHandle_t _handle, cudaStream_t _stream,
            (int *states, size_t n_states, size_t n_reactions, size_t n_species, size_t *fsp_dim,
                   cuda_csr_mat_int stoich, TcoefFun t_func, PropFun prop_func) {

//        cusparse_handle = _handle;
//        stream = _stream;

        std::cout << "n_states = " << n_states << "\n";

        int *d_stoich_vals, *d_stoich_colidxs, *d_stoich_rowptrs;
        cudaMalloc(&d_stoich_vals, stoich.row_ptrs[stoich.n_rows] * sizeof(int));
        cudaMalloc(&d_stoich_colidxs, stoich.row_ptrs[stoich.n_rows] * sizeof(int));
        cudaMalloc(&d_stoich_rowptrs, (stoich.n_rows + 1) * sizeof(int));

        cudaMemcpy(d_stoich_vals, stoich.vals, stoich.row_ptrs[stoich.n_rows] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_stoich_colidxs, stoich.col_idxs, stoich.row_ptrs[stoich.n_rows] * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_stoich_rowptrs, stoich.row_ptrs, (stoich.n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

        std::cout << "Copied stoich to device.\n";

        // Temporary workspace for matrix generation
        int *iwsp;
        cudaMalloc(&iwsp, n_states * sizeof(int));
        CUDACHKERR();

        // Get the max number of threads that can fit to a block
        int max_block_size, num_blocks;

        int device_id;

        cudaGetDevice(&device_id);
        CUDACHKERR();
        cudaDeviceGetAttribute(&max_block_size, cudaDevAttrMaxThreadsPerBlock, device_id);
        CUDACHKERR();
        std::cout << "Max block size = " << max_block_size << "\n";

        num_blocks = (size_t) std::ceil(n_states / (max_block_size * 1.0));

        // Initialize dimensions
        nst = n_states;
        nr = n_reactions;
        ns = n_species;

        std::cout << "nst = " << nst << "nr = " << nr << "ns = " << ns << "\n";

        term.resize(n_reactions);

        // Generate the state space
        fsp_get_states << < num_blocks, max_block_size, n_species * sizeof(size_t) >> >
                                                        (states, n_species, n_states, fsp_dim);
        CUDACHKERR();
        cudaDeviceSynchronize();

        std::cout << "State space generation successful.\n";

        for (size_t ir{0}; ir < nr; ++ir) {
            // Initialize CSR data structure for the ir-th matrix
            cudaMallocManaged((void **) &((term.at(ir)).row_ptrs), (n_states + 1) * sizeof(int));
            CUDACHKERR();
        }

        for (int ir{0}; ir < nr; ++ir) {
            term.at(ir).n_cols = n_states;
            term.at(ir).n_rows = n_states;

            // Count nonzero entries and store off-diagonal col indices to the temporary workspace
            size_t shared_size = ns * sizeof(size_t);
            fspmat_component_get_nnz_per_row << < num_blocks, max_block_size, shared_size >> > (term[ir].row_ptrs +
                                                                                                1, iwsp, states, ir, n_states, n_species, fsp_dim,
                    d_stoich_vals, d_stoich_colidxs, d_stoich_rowptrs);
            CUDACHKERR();
            cudaDeviceSynchronize();
            CUDACHKERR();

            std::cout << "ir = " << ir << " get_nnz_per_row finished.\n";
            // Use cumulative sum to determine the values of the row pointers in CSR

            term[ir].row_ptrs[0] = 0;
            thrust::inclusive_scan(term[ir].row_ptrs, term[ir].row_ptrs + (nst + 1), term[ir].row_ptrs);
            CUDACHKERR();
            std::cout << "ir = " << ir << " inclusive scan finished.\n";

            // Fill the column indices and values
            int nnz;

            cudaMemcpy(&nnz, term[ir].row_ptrs + nst, sizeof(int), cudaMemcpyDeviceToHost); CUDACHKERR();

            std::cout << "nnz = " << nnz << "\n";

            cudaMallocManaged(&(term[ir].vals), nnz * sizeof(double));
            CUDACHKERR();
            cudaMallocManaged(&(term[ir].col_idxs), nnz * sizeof(int));
            CUDACHKERR();

            fspmat_component_fill_data_csr << < num_blocks, max_block_size >> >
                                                            (term[ir].vals, term[ir].col_idxs, term[ir].row_ptrs, nst, ir, iwsp, states, ns,
                                                                    prop_func);
            CUDACHKERR();

            cudaDeviceSynchronize();
            CUDACHKERR();
            std::cout << "ir = " << ir << " fill_data_csr finished.\n";
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

    void FSPMat::action(double t, double *x, double *y) {

    }

    // Destructor
    FSPMat::~FSPMat() {
        for (size_t i{0}; i < nr; ++i) {
            cudaFree(term[i].col_idxs);
            CUDACHKERR();
            cudaFree(term[i].row_ptrs);
            CUDACHKERR();
            cudaFree(term[i].vals);
            CUDACHKERR();
        }
        cusparseDestroyMatDescr(cusparse_descr);
        CUDACHKERR();
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

    __global__

    void
    fspmat_component_get_nnz_per_row(int *nnz_per_row, int *off_indx, int *states, int reaction, size_t n_rows,
                                     size_t n_species, size_t *fsp_bounds,
                                     int *stoich_vals, int *stoich_colidxs, int *stoich_rowptrs) {
        extern __shared__
        size_t wsp[];

        size_t tix = threadIdx.x;
        size_t tid = blockDim.x * blockIdx.x + tix;

        size_t *fsp_bounds_copy = &wsp[0];

        if (tix < n_species)
        {
            fsp_bounds_copy[tix] = fsp_bounds[tix];
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
                off_indx[tid] = state2indx(state, n_species, fsp_bounds_copy);
                nnz_per_row[tid] += 1;
            } else {
                off_indx[tid] = -1;
            }

            reachable_state(state, state, reaction, 1,
                            n_species, stoich_vals, stoich_colidxs, stoich_rowptrs);

//            printf("tid = %d    nnz[tid] = %d \n", (int) tid, nnz_per_row[tid]);
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
                values[i_diag] *= -1.0;

                state = states + dim * off_diag_indx;
                values[i_offdiag] = propensity(state, reaction);

                col_indices[i_diag] = (int) tid;
                col_indices[i_offdiag] = off_diag_indx;
            } else {
                state = states + dim * tid;
                values[rowptr] = propensity(state, reaction);
                values[rowptr] *= -1.0;
                col_indices[rowptr] = (int) tid;
            }
        }
        __syncthreads();
    }
}
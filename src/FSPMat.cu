#include "FSPMat.h"
#include "driver_types.h"

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
            (cusparseHandle_t _handle,
             int *states, size_t n_states, size_t n_reactions, size_t n_species, size_t *fsp_dim,
             cuda_csr_mat_int stoich, TcoefFun t_func, PropFun prop_func) {

        cusparse_handle = _handle;

        cusparseCreateMatDescr(&cusparse_descr);
        cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(cusparse_descr, CUSPARSE_INDEX_BASE_ZERO);

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
        cudaMalloc(&iwsp, n_states * sizeof(int));CUDACHKERR();

        // Get the max number of threads that can fit to a block
        int max_block_size, num_blocks;
        int device_id;

        cudaGetDevice(&device_id);CUDACHKERR();
        cudaDeviceGetAttribute(&max_block_size, cudaDevAttrMaxThreadsPerBlock, device_id);CUDACHKERR();

        num_blocks = (int) std::ceil(n_states / (max_block_size * 1.0));

        // Initialize dimensions
        nst = n_states;
        nr = n_reactions;
        ns = n_species;
        tcoeffunc = t_func;

        term.resize(n_reactions);

        // Generate the state space
        fsp_get_states <<< num_blocks, max_block_size, n_species * sizeof(size_t) >>>
                                                        (states, n_species, n_states, fsp_dim); CUDACHKERR();
        cudaDeviceSynchronize();

        for (size_t ir{0}; ir < nr; ++ir) {
            // Initialize CSR data structure for the ir-th matrix
            cudaMalloc((void **) &((term.at(ir)).row_ptrs), (n_states + 1) * sizeof(int)); CUDACHKERR();
        }

        int h_zero = 0;

        for (int ir{0}; ir < nr; ++ir) {
            term.at(ir).n_cols = n_states;
            term.at(ir).n_rows = n_states;

            // Count nonzero entries and store off-diagonal col indices to the temporary workspace
            fspmat_component_get_nnz_per_row << < num_blocks, max_block_size, ns*sizeof(size_t) >> > (term[ir].row_ptrs +
                                                                                                1, iwsp, states, ir, n_states, n_species, fsp_dim,
                    d_stoich_vals, d_stoich_colidxs, d_stoich_rowptrs);CUDACHKERR();
            cudaDeviceSynchronize();CUDACHKERR();

            // Use cumulative sum to determine the values of the row pointers in CSR

            cudaMemcpy(term[ir].row_ptrs, &h_zero, sizeof(int), cudaMemcpyHostToDevice); CUDACHKERR();
            thrust::device_ptr<int> ptr(term[ir].row_ptrs);
            thrust::inclusive_scan(ptr, ptr + (nst + 1), ptr);CUDACHKERR();
            cudaDeviceSynchronize(); CUDACHKERR();

            // Fill the column indices and values
            int nnz;
            cudaMemcpy(&nnz, term[ir].row_ptrs + nst, sizeof(int), cudaMemcpyDeviceToHost);CUDACHKERR();
            term[ir].nnz = (size_t) nnz;

            cudaMalloc((void**) &(term[ir].vals), nnz * sizeof(double));CUDACHKERR();
            cudaMalloc((void**) &(term[ir].col_idxs), nnz * sizeof(int));CUDACHKERR();

            fspmat_component_fill_data_csr << < num_blocks, max_block_size >> >
                                                            (term[ir].vals, term[ir].col_idxs, term[ir].row_ptrs, nst, ir, iwsp, states, ns,
                                                                    prop_func); CUDACHKERR();


            cudaDeviceSynchronize();CUDACHKERR();
        }

        cudaFree(d_stoich_colidxs); CUDACHKERR();
        cudaFree(d_stoich_rowptrs);CUDACHKERR();
        cudaFree(d_stoich_vals); CUDACHKERR();
        cudaFree(iwsp); CUDACHKERR();
    }

    void FSPMat::action(double t, thrust_dvec& x, thrust_dvec& y) {
        tcoef = tcoeffunc(t);

        assert(x.size() == nst);
        assert(y.size() == nst);

        const double h_one = 1.0;

//        thrust::fill(y.begin(), y.end(), 0.0); CUDACHKERR();
        cudaDeviceSynchronize(); CUDACHKERR();

        cusparseStatus_t stat;

        for (size_t i{0}; i < nr; ++i){
            stat = cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, (int) nst, (int) nst, (int) term[i].nnz,
                              (double*) &tcoef[i], cusparse_descr,
                              term[i].vals, term[i].row_ptrs, term[i].col_idxs,
                              (double*) thrust::raw_pointer_cast(&x[0]),
                              &h_one,
                              (double*) thrust::raw_pointer_cast(&y[0]));
            assert (stat == CUSPARSE_STATUS_SUCCESS);

            cudaDeviceSynchronize(); CUDACHKERR();
        }
    }

    // Destructor
    void FSPMat::destroy(){
        for (size_t i{0}; i < nr; ++i) {
            if (term[i].col_idxs) {
                cudaFree(term[i].col_idxs); CUDACHKERR();
            }
            if (term[i]. row_ptrs) {
                cudaFree(term[i].row_ptrs); CUDACHKERR();
            }
            if (term[i].vals) {
                cudaFree(term[i].vals); CUDACHKERR();
            }
            term[i].col_idxs = nullptr;
            term[i].row_ptrs = nullptr;
            term[i].vals = nullptr;
        }
        if (cusparse_descr) {
            cusparseDestroyMatDescr(cusparse_descr); CUDACHKERR();
        }
    }

    FSPMat::~FSPMat() {
        destroy();
    }
}
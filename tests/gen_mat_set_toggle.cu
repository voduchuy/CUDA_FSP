#include<cuda_runtime.h>
#include<cuda.h>
#include<cusparse.h>
#include<cmath>
#include<time.h>
#include<iostream>
#include<iomanip>
#include<thrust/scan.h>
#include<string>

#define CUDACHKERR() { \
cudaError_t ierr = cudaGetLastError();\
if (ierr != cudaSuccess){ \
    printf("%s in %s at line %d\n", cudaGetErrorString(ierr), __FILE__, __LINE__);\
    exit(EXIT_FAILURE); \
}\
}\

/* TEST USING CUDA TO GENERATE DATA FOR AN ELL MATRIX REPRESENTING THE TOGGLE CME
 *
 *
 * */

__host__ __device__
void reachable_state(int *x, int *rx, int reaction, int direction = 1) {
    assert( (direction == 1) || (direction == -1));

    switch (reaction) {
        case 0:
            rx[0] = x[0] + direction * 1;
            rx[1] = x[1];
            break;
        case 1:
            rx[0] = x[0] - direction * 1;
            rx[1] = x[1];
            break;
        case 2:
            rx[0] = x[0];
            rx[1] = x[1] + direction * 1;
            break;
        case 3:
            rx[0] = x[0];
            rx[1] = x[1] - direction * 1;
            break;
    }
}

__host__ __device__

double propensity(int *x, int reaction) {
    double prop_val;
    switch (reaction) {
        case 0:
            prop_val = 1.0 / (1.0 + std::pow(1.0 * x[1], 2.0));
            break;
        case 1:
            prop_val = 1.0 * x[0];
            break;
        case 2:
            prop_val = 1.0 / (1.0 + std::pow(1.0 * x[0], 2.0));
            break;
        case 3:
            prop_val = 1.0 * x[1];
            break;
    }
    return prop_val;
}

__host__ __device__

void indx2state(size_t indx, int *state, size_t dim, size_t *fsp_bounds) {
    for (size_t i{1}; i <= dim; i++) {
        state[i - 1] = indx % (fsp_bounds[i - 1] + 1);
        indx = indx / (fsp_bounds[i - 1] + 1);
    }
}

__host__ __device__

void state2indx(int *state, int &indx, size_t dim, size_t *fsp_bounds) {
    indx = 0;
    int nprod = 1;
    for (size_t i{1}; i <= dim; ++i) {
        indx += state[i-1] * nprod;
        nprod *= (fsp_bounds[i-1] + 1);
    }
}

__global__
void get_states(int *d_states, size_t dim, size_t n_states, size_t *n_bounds) {
    extern __shared__ size_t n_bounds_copy[];

    size_t ti = threadIdx.x;
    size_t indx = blockIdx.x * blockDim.x + ti;

    if (ti < dim)
        n_bounds_copy[ti] = n_bounds[ti];

    __syncthreads();

    if (indx < n_states) {
        indx2state(indx, &d_states[indx*dim], dim, &n_bounds_copy[0]);
    }
}

__global__
void
cme_component_get_nnz_per_row(int *nnz_per_row, int *off_indx, int reaction, size_t n_rows,
                              size_t dim, int* states, size_t *fsp_bounds) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ size_t fsp_bounds_copy[];

    for (size_t k{0}; k < dim; ++k) {
        fsp_bounds_copy[k] = fsp_bounds[k];
    }

    int *state;


    if (tid < n_rows) {
        state = &states[tid*dim];

        indx2state(tid, &state[0], dim, fsp_bounds_copy);
        reachable_state(state, state, reaction, -1);

        bool reachable = true;
        for (size_t k{0}; k < dim; ++k) {
            if ((state[k] < 0) || (state[k] > fsp_bounds_copy[k])) {
                reachable = false;
//                break;
            }
        }

        nnz_per_row[tid] = 1;
        if (reachable) {
            state2indx(state, off_indx[tid], dim, fsp_bounds_copy);
            nnz_per_row[tid] += 1;
        } else {
            off_indx[tid] = -1;
        }

        reachable_state(state, state, reaction, 1);
    }
}

__global__
void
cme_component_fill_data_csr(double* values, int* col_indices, int* row_ptrs, size_t n_rows, int reaction,
                            int* off_diag_indices, int* states, size_t dim){
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    int off_diag_indx, rowptr, i_diag, i_offdiag;
    int *state;

    if (tid < n_rows){
        state = states + dim*tid;
        off_diag_indx = off_diag_indices[tid];
        rowptr = row_ptrs[tid];

        if (off_diag_indx >= 0){
            if (off_diag_indx > tid)
            {
                i_diag = rowptr;
                i_offdiag = rowptr+1;
            }
            else
            {
                i_diag = rowptr+1;
                i_offdiag = rowptr;
            }

            values[i_diag] = propensity(state, reaction);
            col_indices[i_diag] = (int) tid;
            values[i_diag] *= -1.0;

            reachable_state(state, state, reaction, -1);
            values[i_offdiag] = propensity(state, reaction);
            col_indices[i_offdiag] = off_diag_indx;
            reachable_state(state, state, reaction, 1);
        }
        else
        {
            values[rowptr] = propensity(state, reaction);
            values[rowptr] *= -1.0;
            col_indices[rowptr] = (int) tid;
        }

    }
}


int main(int argc, char *argv[]) {
    clock_t t1, t2;

    size_t dim = 2;
    int reaction = 1;

    size_t blockSize = 1024;

    for (size_t i{0}; i < argc; ++i) {
        if (strcmp(argv[i], "-block_size") == 0) {
            blockSize = (size_t) std::stoi(argv[i + 1]);
            i += 1;
        }
        i += 1;
    }

    size_t *n_bounds;
    int *states;
    int *row_pointers, *col_indices, *int_workspace;
    double *values;

    cudaMallocManaged(&n_bounds, dim*sizeof(size_t));

    n_bounds[0] = (1 << 12) - 1;
    n_bounds[1] = (1 << 14) - 1;

    std::cout << n_bounds[0] << " " << n_bounds[1] << "\n";

    size_t n_states = 1;
    for (size_t i{0}; i < dim; ++i) {
        n_states *= (n_bounds[i] + 1);
    }
    std::cout << "Total number of states:" << n_states << "\n";


    cudaMallocManaged((void **) &row_pointers, (n_states+1) * sizeof(int));
    CUDACHKERR();
    cudaMallocManaged((void **) &int_workspace, n_states * sizeof(int));
    CUDACHKERR();


    t1 = clock();
    cudaMallocManaged(&states, n_states * dim * sizeof(int));CUDACHKERR();
    get_states << < (size_t) std::ceil(n_states / (1024.0)), 1024, dim*sizeof(size_t) >> > (states, dim, n_states, n_bounds);
    CUDACHKERR();
    cudaDeviceSynchronize();
    t2 = clock();
    std::cout << "Generate states with GPU take " << (double) (t2 - t1) / CLOCKS_PER_SEC * 1000.0 << " ms.\n";

    t1 = clock();
    cme_component_get_nnz_per_row <<< std::ceil(n_states / (blockSize * 1.0)), blockSize, dim*sizeof(size_t) >>>
                                                                                (row_pointers+1, int_workspace, reaction, n_states, dim, states, n_bounds);
    CUDACHKERR();
    cudaDeviceSynchronize();
    t2 = clock();
    std::cout << "Running kernel takes " << (float) (t2 - t1)/CLOCKS_PER_SEC*1000.0 << " ms.\n";

    for (size_t i{0}; i < 50; ++i)
    {
        std::cout << "nnz(" << i << ") = " << row_pointers[i+1] << ", off_col_indx("<<i<<") = "<<int_workspace[i] << "\n.";
    }

    row_pointers[0] = 0;
    thrust::inclusive_scan(row_pointers, row_pointers + (n_states + 1), row_pointers);

    for (size_t i{0}; i < 50; ++i)
    {
        std::cout << "row_pointers(" << i << ") = " << row_pointers[i] << ", off_col_indx("<<i<<") = "<<int_workspace[i] << "\n.";
    }
    std::cout << "row_pointers(" << n_states <<") = " << row_pointers[n_states] << "\n";

    cudaMallocManaged(&values, row_pointers[n_states]*sizeof(double));
    CUDACHKERR();
    cudaMallocManaged(&col_indices, row_pointers[n_states]*sizeof(int));
    CUDACHKERR();

    t1 = clock();
    cme_component_fill_data_csr<<< std::ceil(n_states / (blockSize * 1.0)), blockSize>>>(values,col_indices, row_pointers, n_states, reaction, int_workspace, states, dim);
    CUDACHKERR();
    cudaDeviceSynchronize();
    t2 = clock();
    std::cout << "Filling CSR data took " << (float) (t2 - t1)/CLOCKS_PER_SEC*1000.0 << " ms.\n";

    std::cout << std::setw(5) << "Row " << std::setw(5) << "Col" << std::setw(5) <<"Val" << std::setw(10) << "State \n";

    for (size_t i{0}; i < 50; ++i)
    {
        for (size_t j{row_pointers[i]}; j < row_pointers[i+1]; ++j){
            std::cout << std::setw(5) << i << " " << std::setw(5) << col_indices[j] << " " << std::setw(5) << values[j]
                      << std::setw(5) << states[i*dim] << std::setw(5) << states[i*dim+1] << "\n";
        }
    }

    cudaFree(col_indices);
    CUDACHKERR();
    cudaFree(values);
    CUDACHKERR();
    cudaFree(states);
    CUDACHKERR();
    cudaFree(row_pointers);
    CUDACHKERR();
    cudaFree(int_workspace);
    CUDACHKERR();
    cudaFree(n_bounds);
    CUDACHKERR();
    return 0;
}
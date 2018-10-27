#include<cuda_runtime.h>
#include<cuda.h>
#include<cusparse.h>
#include<cmath>
#include<time.h>
#include<iostream>
#include<thrust/scan.h>

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

void reachable_from(int *x, int *rx, int reaction) {
    switch (reaction) {
        case 0:
            rx[0] = x[0] - 1;
            rx[1] = x[1];
            break;
        case 1:
            rx[0] = x[0] + 1;
            rx[1] = x[1];
            break;
        case 2:
            rx[0] = x[0];
            rx[1] = x[1] - 1;
            break;
        case 3:
            rx[0] = x[0];
            rx[1] = x[1] + 1;
            break;
    }
}

__host__ __device__

void propensity(int *x, double &prop_val, int reaction) {
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
}

__host__ __device__

void indx2state(size_t indx, int *state, int dim, size_t *fsp_bounds) {
    for (size_t i{1}; i <= dim; i++) {
        state[i - 1] = indx % (fsp_bounds[i - 1] + 1);
        indx = indx / (fsp_bounds[i - 1] + 1);
    }
}

__host__ __device__

void state2indx(int *state, int &indx, int dim, size_t *fsp_bounds) {
    indx = state[0];
    for (size_t i{1}; i < dim; ++i) {
        indx += state[i] * (fsp_bounds[i - 1] + 1);
    }
}

__global__
void
cme_component_get_nnz_per_row(int *nnz_per_row, int *off_indx, int reaction, int n_rows, int dim, size_t *fsp_bounds) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    size_t *fsp_bounds_copy;
    fsp_bounds_copy = new size_t[dim];

    for (size_t k{0}; k < dim; ++k) {
        fsp_bounds_copy[k] = fsp_bounds[k];
    }

    int *state;
    int *rstate;
    state = new int[dim];
    rstate = new int[dim];

    if (tid < n_rows) {

        indx2state(tid, &state[0], dim, fsp_bounds_copy);
        reachable_from(state, rstate, reaction);

        bool reachable = true;
        for (size_t k{0}; k < dim; ++k) {
            if ((rstate[k] < 0) || (rstate[k] > fsp_bounds_copy[k])) {
                reachable = false;
                break;
            }
        }

        nnz_per_row[tid] = 1;
        if (reachable) {
            state2indx(rstate, off_indx[tid], dim, fsp_bounds_copy);
            nnz_per_row[tid] += 1;
        } else {
            off_indx[tid] = -1;
        }
    }
    delete[] fsp_bounds_copy;
    delete[] state;
    delete[] rstate;
}


int main(int argc, char *argv[]) {
    clock_t t1, t2;

    size_t dim = 2;

    size_t blockSize = 1024;

    std::cout << sizeof(int) << "\n" << sizeof(double) << "\n";

    for (size_t i{0}; i < argc; ++i) {
        if (strcmp(argv[i], "-block_size") == 0) {
            blockSize = (size_t) std::stoi(argv[i + 1]);
            i += 1;
        }
        i += 1;
    }

    size_t *n_bounds = new size_t[dim];
    n_bounds[0] = 1 << 7;
    n_bounds[1] = 1 << 7;

    size_t *d_n_bounds;
    cudaMalloc((void **) &d_n_bounds, dim * sizeof(size_t));
    CUDACHKERR();
    cudaMemcpy(d_n_bounds, n_bounds, dim, cudaMemcpyHostToDevice);
    CUDACHKERR();

    size_t n_states = 1;
    for (size_t i{0}; i < dim; ++i) {
        n_states *= (n_bounds[i] + 1);
    }
    std::cout << "Total number of states:" << n_states << "\n";

    int *nnz_per_row, *off_col_indx;

    cudaMallocManaged((void **) &nnz_per_row, n_states * sizeof(int));
    CUDACHKERR();
    cudaMallocManaged((void **) &off_col_indx, n_states * sizeof(int));
    CUDACHKERR();

    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(nnz_per_row, n_states*sizeof(int), device, NULL);
    cudaMemPrefetchAsync(off_col_indx, n_states*sizeof(int), device, NULL);

    t1 = clock();
    cme_component_get_nnz_per_row << < std::ceil(n_states / (blockSize * 1.0)), blockSize >> >
                                                                                (nnz_per_row, off_col_indx, 0, n_states, dim, d_n_bounds);
    CUDACHKERR();
    cudaDeviceSynchronize();
    t2 = clock();
    std::cout << "Running kernel takes " << (float) (t2 - t1)/CLOCKS_PER_SEC*1000.0 << " ms.\n";

    for (size_t i{0}; i < 50; ++i)
    {
        std::cout << "nnz(" << i << ") = " << nnz_per_row[i] << ", off_col_indx("<<i<<") = "<<off_col_indx[i] << "\n.";
    }

    cudaFree(nnz_per_row);
    CUDACHKERR();
    cudaFree(off_col_indx);
    CUDACHKERR();
    cudaFree(d_n_bounds);
    CUDACHKERR();
    delete[] n_bounds;
    return 0;
}
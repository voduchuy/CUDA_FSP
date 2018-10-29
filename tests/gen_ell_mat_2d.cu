#include<cuda_runtime.h>
#include<cuda.h>
#include<cusparse.h>
#include<cmath>
#include<time.h>
#include<iostream>

#define cudachkerr(ierr) \
if (ierr != cudaSuccess){ \
    printf("%s in %s at line %d\n", cudaGetErrorString(ierr), __FILE__, __LINE__);\
    exit(EXIT_FAILURE); \
}\

/* TEST USING CUDA TO GENERATE DATA FOR AN ELL MATRIX REPRESENTING THE TOGGLE CME
 *
 *
 * */

__host__ __device__
void reachable_state(int *x, int *rx, int reaction, int direction = 1) {
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
void indx2state(size_t indx, int *state, int dim, int *fsp_bounds) {
    for (size_t i{1}; i <= dim; i++) {
        state[i - 1] = indx % (fsp_bounds[i - 1] + 1);
        indx = indx / (fsp_bounds[i - 1] + 1);
    }
}

__host__ __device__
void state2indx(int *state, int &indx, int dim, int *fsp_bounds) {
    indx = state[0];
    for (size_t i{1}; i < dim; ++i) {
        indx += state[i] * (fsp_bounds[i - 1] + 1);
    }
}

__global__
void get_states(int *d_states, size_t dim, int n_states, int *n_bounds) {
    extern __shared__ int n_bounds_copy[];

    size_t ti = threadIdx.x;
    size_t indx = blockIdx.x * blockDim.x + ti;

    if (ti < dim)
        n_bounds_copy[ti] = n_bounds[ti];

    __syncthreads();

    if (indx < n_states) {
        indx2state(indx, &d_states[indx * dim], dim, &n_bounds_copy[0]);
    }
}

__global__
void fill_ell_matrix(double *val, int *offset, int max_nnz_per_row, int n_rows, int dim,
                     int *states, int *fsp_bounds) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ int fsp_bounds_copy[];

    if (threadIdx.x < dim)
        fsp_bounds_copy[threadIdx.x] = fsp_bounds[threadIdx.x];

    int *state;

    if (tid < n_rows) {
        state = &states[tid * dim];

        /* First phase: fill the off-diagonals */
        indx2state(tid, &state[0], dim, fsp_bounds_copy);

        for (size_t i{0}; i < max_nnz_per_row - 1; ++i) {
            reachable_state(state, state, i, -1);

            bool reachable = true;
            for (size_t k{0}; k < dim; ++k) {
                if ((state[k] < 0) || (state[k] > fsp_bounds_copy[k])) {
                    reachable = false;
                    break;
                }
            }
            if (reachable) {
                state2indx(state, offset[max_nnz_per_row * tid + i + 1], dim, fsp_bounds_copy);
                propensity(state, val[max_nnz_per_row * tid + i + 1], i);
            } else {
                offset[max_nnz_per_row * tid + i + 1] = -1;
                val[max_nnz_per_row * tid + i + 1] = 0.0;
            }

            // Return state value to the original
            reachable_state(state, state, i, 1);
        }

        __syncthreads();

        /* Second phase: fill the diagonals */
        val[max_nnz_per_row * tid] = 0.0;
        offset[max_nnz_per_row * tid] = (int) tid;
        double tmp;
        for (size_t i{0}; i < max_nnz_per_row - 1; ++i) {
            propensity(state, tmp, i);
            val[max_nnz_per_row * tid] += tmp;
            __syncthreads();
        }
        val[max_nnz_per_row * tid] *= -1.0;

    }
}

void fill_ell_matrix_cpu(double *val, int *offset, int max_nnz_per_row, int n_rows, int dim, int *fsp_bounds) {

    int *state;
    int *rstate;

    state = new int[dim];
    rstate = new int[dim];

    for (size_t tid{0}; tid < n_rows; ++tid) {

        /* First phase: fill the off-diagonals */

        indx2state(tid, &state[0], dim, fsp_bounds);

        for (size_t i{0}; i < max_nnz_per_row - 1; ++i) {
            reachable_state(state, rstate, i, -1);

            bool reachable = true;
            for (size_t k{0}; k < dim; ++k) {
                if ((rstate[k] < 0) || (rstate[k] > fsp_bounds[k])) {
                    reachable = false;
                    break;
                }
            }
            if (reachable) {
                state2indx(rstate, offset[max_nnz_per_row * tid + i + 1], dim, fsp_bounds);
                propensity(rstate, val[max_nnz_per_row * tid + i + 1], i);
            } else {
                offset[max_nnz_per_row * tid + i + 1] = -1;
                val[max_nnz_per_row * tid + i + 1] = 0.0;
            }
        }

        /* Second phase: fill the diagonals */
        val[max_nnz_per_row * tid] = 0.0;
        offset[max_nnz_per_row * tid] = (int) tid;
        double tmp;
        for (size_t i{0}; i < max_nnz_per_row - 1; ++i) {
            propensity(state, tmp, i);
            val[max_nnz_per_row * tid] += tmp;
        }
        val[max_nnz_per_row * tid] *= -1.0;


    }
    delete[] rstate;
    delete[] state;
}

int main(int argc, char *argv[]) {
    std::cout << "Running from AWS!\n";

    cudaError_t cuerr;
    clock_t t1, t2;

    size_t dim = 2;

    size_t blockSize = 1024;

    for (size_t i{0}; i < argc; ++i) {
        if (strcmp(argv[i], "-block_size") == 0) {
            blockSize = (size_t) std::stoi(argv[i + 1]);
            i += 1;
        }
        i += 1;
    }

    int *n_bounds = new int[dim];
//    n_bounds[0] = 1;
//    n_bounds[1] = 1;
    n_bounds[0] = 1 << 12;
    n_bounds[1] = 1 << 12;

    int *d_n_bounds;
    cuerr = cudaMalloc((void **) &d_n_bounds, dim * sizeof(int));
    cudachkerr(cuerr);
    cuerr = cudaMemcpy(d_n_bounds, n_bounds, dim * sizeof(int), cudaMemcpyHostToDevice);
    cudachkerr(cuerr);

    size_t n_states = 1;
    for (size_t i{0}; i < dim; ++i) {
        n_states *= (n_bounds[i] + 1);
    }
    std::cout << "Total number of states:" << n_states << "\n";

    double *d_a_val;
    int *d_a_offset;
    int *d_states;

    cuerr = cudaMallocManaged((void **) &d_a_val, n_states * 5 * sizeof(double));
    cudachkerr(cuerr);
    cuerr = cudaMallocManaged((void **) &d_a_offset, n_states * 5 * sizeof(int));
    cudachkerr(cuerr);

    t1 = clock();
    cuerr = cudaMallocManaged(&d_states, n_states * dim * sizeof(int));
    cudachkerr(cuerr);
    get_states << < (size_t) std::ceil(n_states / (1024.0)), 1024, dim*sizeof(int) >> > (d_states, dim, n_states, d_n_bounds);
    cuerr = cudaPeekAtLastError();
    cudachkerr(cuerr);
    cudaDeviceSynchronize();
    t2 = clock();
    std::cout << "Generate states with GPU take " << (double) (t2 - t1) / CLOCKS_PER_SEC * 1000.0 << " ms.\n";

    t1 = clock();
    size_t sharedMem = dim * sizeof(int);
    fill_ell_matrix << < std::ceil(n_states / (blockSize * 1.0)), blockSize, sharedMem >> >
                                                                             (d_a_val, d_a_offset, 5, n_states, dim,
                                                                                     d_states, d_n_bounds);
    cuerr = cudaPeekAtLastError();
    cudachkerr(cuerr);
    cudaDeviceSynchronize();
    t2 = clock();

    std::cout << "Generate matrix with GPU takes " << (double) (t2 - t1) / CLOCKS_PER_SEC * 1000.0 << " ms.\n";

    for (size_t i{0}; i < 10; ++i) {
        for (size_t k{0}; k < 5; ++k) {
            printf("%.4f ", d_a_val[i * 5 + k]);
        }
        printf("\n");
    }

    for (size_t i{0}; i < 10; ++i) {
        for (size_t k{0}; k < 5; ++k) {
            printf("%d ", d_a_offset[i * 5 + k]);
        }
        printf("\n");
    }

//    double *a_val;
//    int *a_offset;
//
//    a_val = new double[n_states * 5 * sizeof(double)];
//    a_offset = new int[n_states * 5 * sizeof(int)];

    t1 = clock();
    fill_ell_matrix_cpu(d_a_val, d_a_offset, 5, n_states, dim, n_bounds);
    t2 = clock();
    std::cout << "Generate matrix with CPU takes " << (double) (t2 - t1) / CLOCKS_PER_SEC * 1000.0 << " ms.\n";

    for (size_t i{0}; i < 10; ++i) {
        for (size_t k{0}; k < 5; ++k) {
            printf("%.4f ", d_a_val[i * 5 + k]);
        }
        printf("\n");
    }

    for (size_t i{0}; i < 10; ++i) {
        for (size_t k{0}; k < 5; ++k) {
            printf("%d ", d_a_offset[i * 5 + k]);
        }
        printf("\n");
    }

    cudaFree(d_states);
    cudaFree(d_a_val);
    cudaFree(d_a_offset);
    cudaFree(d_n_bounds);
//
//    delete[] a_val;
//    delete[] a_offset;
//    delete[] n_bounds;
    return 0;
}
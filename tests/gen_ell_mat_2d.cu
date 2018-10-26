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

__device__
void reachable_from(int* x, int* rx, int reaction){
    switch (reaction){
        case 0:
            rx[0] = x[0]+1;
            rx[1] = x[1];
            break;
        case 1:
            rx[0] = x[0]-1;
            rx[1] = x[1];
            break;
        case 2:
            rx[0] = x[0];
            rx[1] = x[1] + 1;
            break;
        case 3:
            rx[0] = x[0];
            rx[1] = x[1] - 1;
            break;
    }
}

__device__
void propensity(int* x, double* prop){
   prop[0] = 1.0/(1.0 + std::pow(1.0*x[1], 2.0));
   prop[1] = 1.0*x[0];
   prop[2] = 1.0/(1.0 + std::pow(1.0*x[0], 2.0));
   prop[3] = 1.0*x[1];
}

__device__
void indx2state(size_t indx, int* state, int dim, size_t* fsp_bounds)
{
    for (size_t i {1}; i <= dim; i++)
    {
        state[i-1] = indx%(fsp_bounds[i-1] + 1);
        indx = indx/(fsp_bounds[i-1] + 1);
    }
}

__device__
void state2indx(int* state, int& indx, int dim, size_t* fsp_bounds)
{
    indx = state[0];
    for (size_t i {1}; i < dim; ++i)
    {
        indx += state[i]*(fsp_bounds[i-1]+1);
    }
}

__global__
void fill_ell_matrix(double* val, int* offset, int max_nnz_per_row, int n_rows, int dim, size_t* fsp_bounds)
{
    size_t tid = threadIdx.x;

    if (tid < n_rows)
    {
        size_t* fsp_bounds_copy;
        int* state;
        int* rstate;
        double* prop_vals;

        state = new int[dim];
        rstate = new int[dim];
        prop_vals = new double[max_nnz_per_row-1];
        fsp_bounds_copy = new size_t[dim];
        for (size_t k{0}; k < dim; ++k)
        {
            fsp_bounds_copy[k] = fsp_bounds[k];
        }

        double diag_val;

        indx2state(tid, &state[0], dim, fsp_bounds_copy);
        propensity(&state[0], &prop_vals[0]);
        diag_val = 0.0;

        for (size_t i{1}; i < max_nnz_per_row;++i){
            diag_val += prop_vals[i-1];

            val[max_nnz_per_row*tid + i] = prop_vals[i-1];
        }
        val[max_nnz_per_row*tid] = -1.0*diag_val;

        // fill the offset
        offset[max_nnz_per_row*tid] = (int) tid;
        for (size_t i{0}; i < max_nnz_per_row-1; ++i){
            reachable_from(state, rstate, i);

            bool reachable = true;
            for (size_t k{0}; k < dim; ++k){
                if ((rstate[k] < 0) || (rstate[k] > fsp_bounds_copy[k])){
                    reachable = false;
                    break;
                }
            }
            if (reachable)
            {
                state2indx(rstate, offset[max_nnz_per_row*tid + i + 1], dim, fsp_bounds_copy);
            }
            else
            {
                offset[max_nnz_per_row*tid + i + 1] = -1;
            }
        }

        delete[] rstate;
        delete[] state;
        delete[] prop_vals;
    }
}

int main(int argc, char* argv[]) {
    cudaError_t cuerr;
    clock_t t1, t2;

    size_t dim = 2;

    size_t blockSize = (size_t) std::stoi(argv[1]);

    size_t *n_bounds = new size_t[dim];
//    n_bounds[0] = 1;
//    n_bounds[1] = 1;
    n_bounds[0] = 1<<15; n_bounds[1] = 1<<15;

    size_t *d_n_bounds;
    cuerr = cudaMalloc((void **) &d_n_bounds, dim * sizeof(size_t));
    cudachkerr(cuerr);
    cuerr = cudaMemcpy(d_n_bounds, n_bounds, dim * sizeof(size_t), cudaMemcpyHostToDevice);
    cudachkerr(cuerr);

    size_t n_states = 1;
    for (size_t i{0}; i < dim; ++i) {
        n_states *= (n_bounds[i] + 1);
    }
    std::cout << "Total number of states:" << n_states << "\n";

    double *d_a_val;
    int *d_a_offset;

    cuerr = cudaMallocManaged((void **) &d_a_val, n_states * 5 * sizeof(double));
    cudachkerr(cuerr);
    cuerr = cudaMallocManaged((void **) &d_a_offset, n_states * 5 * sizeof(int));
    cudachkerr(cuerr);

    t1 = clock();
    fill_ell_matrix << < std::ceil(n_states / (blockSize * 1.0)), blockSize >> >
                                                                  (d_a_val, d_a_offset, 5, n_states, dim, d_n_bounds);
    cuerr = cudaPeekAtLastError();
    cudachkerr(cuerr);
    cudaDeviceSynchronize();
    t2 = clock();

    std::cout << "Generate matrix takes " << (double) (t2 - t1) / CLOCKS_PER_SEC * 1000.0 << " ms.\n";

//    if (n_states < 10)
//    {
//    double *a_val;
//    int *a_offset;
//
//    a_val = new double[n_states * 5 * sizeof(double)];
//    a_offset = new int[n_states * 5 * sizeof(int)];

//    cudaMemcpy(a_val, d_a_val, 5 * sizeof(double) * n_states, cudaMemcpyDeviceToHost);
//    cudaMemcpy(a_offset, d_a_offset, 5 * sizeof(int) * n_states, cudaMemcpyDeviceToHost);

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

//    delete[] a_val;
//    delete[] a_offset;
//}
    cudaFree(d_a_val);
    cudaFree(d_a_offset);
    cudaFree(d_n_bounds);
    delete[] n_bounds;
    return 0;
}
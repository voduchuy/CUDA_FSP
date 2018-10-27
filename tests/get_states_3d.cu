//
// Created by huy on 10/24/18.
//
#include<cuda_runtime.h>
#include<cuda.h>
#include<iostream>
#include "../../../../../usr/local/cuda/include/driver_types.h"
#include<time.h>

#define cudachkerr(ierr) \
if (ierr != cudaSuccess){ \
    printf("%s in %s at line %d\n", cudaGetErrorString(ierr), __FILE__, __LINE__);\
    exit(EXIT_FAILURE); \
}\

__host__ __device__
void indx2state(size_t indx, int* state, int dim, size_t* n_bounds)
{
    for (size_t i {1}; i <= dim; i++)
    {
        state[i-1] = indx%(n_bounds[i-1] + 1);
        indx = indx/(n_bounds[i-1] + 1);
    }
}

__global__
void get_states(int* d_states, size_t dim, int n_states, size_t* n_bounds)
{
    extern __shared__ size_t n_bounds_copy[];
//    n_bounds_copy = new size_t[dim];

    size_t ti = threadIdx.x;
    size_t indx = blockIdx.x*blockDim.x + ti;

    if (ti < dim)
        n_bounds_copy[ti] = n_bounds[ti];

    __syncthreads();

    if (indx < n_states)
    {
        indx2state(indx, &d_states[indx*dim], dim, &n_bounds_copy[0]);
    }

//    delete[] n_bounds_copy;
}

void get_states_cpu(int* states, size_t dim, int n_states, size_t* n_bounds)
{
    for (size_t indx{0}; indx < n_states; ++indx)
    {
        indx2state(indx, &states[indx*dim], dim, n_bounds);
//        std::cout << states[indx*dim] << std::endl;
    }
}

int main()
{
    cudaError_t cuerr;
    clock_t t1, t2;

    size_t d = 3;

    size_t* n_bounds = new size_t[d];
    n_bounds[0] = 1023; n_bounds[1] = 1023; n_bounds[2] = 1023;

    size_t* d_n_bounds;
    cudaMalloc((void**) &d_n_bounds, d*sizeof(size_t));
    cudaMemcpy(d_n_bounds, n_bounds, d*sizeof(size_t), cudaMemcpyHostToDevice);

    size_t n_states = 1;
    for (size_t i{0}; i < d; ++i){
        n_states *= (n_bounds[i] + 1);
    }
    std::cout << "Total number of states:" << n_states << "\n";

    int *d_states, *states, *states_cpu;

    states = new int[n_states*d];
    states_cpu = new int[n_states*d];

    cuerr = cudaMallocManaged((void **) &d_states, d*n_states*sizeof(int));
    cudachkerr(cuerr);

    t1 = clock();
    get_states_cpu(states_cpu, d, n_states, n_bounds);
    t2 = clock();
    std::cout << "Generate states with CPU take "<< (double) (t2-t1)/CLOCKS_PER_SEC*1000.0 << " ms.\n";

    t1 = clock();
    get_states<<< (size_t) std::ceil(n_states/(32.0)), 32, 1024 >>>(d_states, d, n_states, d_n_bounds);
    cuerr = cudaPeekAtLastError();
    cudachkerr(cuerr);
    cudaDeviceSynchronize();
    t2 = clock();
    std::cout << "Generate states with GPU take "<< (double) (t2-t1)/CLOCKS_PER_SEC*1000.0 << " ms.\n";

    cuerr = cudaMemcpy((void*) states, (void*) d_states, d*n_states*sizeof(int), cudaMemcpyDeviceToHost);
    cudachkerr(cuerr);

    cudaDeviceSynchronize();

    bool success = true;
    for (size_t i{0}; i < n_states; ++i)
    {
        for (size_t k{0}; k < d; ++k)
        {
            if (states[i*d + k] != states_cpu[i*d + k])
            {
                success = false;
//                std::cout << "Fail\n";
                break;
            }
        }
    }

    std::cout << "Success: " << success << " \n";

    cudaFree(d_n_bounds);
    cudaFree(d_states);
    delete[] states_cpu;
    delete[] states;
    delete[] n_bounds;
    return 0;
}



#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <device_launch_parameters.h>

__global__ void vectorAdd(int *a, int *b, int* c)
{
    int i = threadIdx.x; // Create a list of threads, x represent the number of the vector that we are in
    c[i] = a[i] + b[i];

    return;
}

int main ()
{
    int a [] = {1,2,3};
    int b [] = {4,6,3};

    int c [sizeof(a) / sizeof(int)] = {0};


    int * cudaA = 0;
    int * cudaB = 0;
    int * cudaC = 0;

    // Allocate memory on the GPU
    cudaMalloc(&cudaA, sizeof(a));
    cudaMalloc(&cudaB, sizeof(a));
    cudaMalloc(&cudaC, sizeof(a));

    cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaC, c, sizeof(c), cudaMemcpyHostToDevice);


    vectorAdd<<<1, sizeof(a) / sizeof(int) >>> (cudaA, cudaB, cudaC);
    // Grid with 1 block, c number of threads

    cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);

    for (int i = 0; i< 3; ++i){
        std::cout<< c[i];
    }
    return;

}
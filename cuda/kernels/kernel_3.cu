#include <cuda_runtime.h>
#include <iostream>

// Define the block size (number of threads per block in each dimension)
#define BLOCKSIZE 16

// CUDA kernel for matrix multiplication with shared memory
__global__ void matrixMulKernel(float *A, float *B, float *C, int M, int K, int N, float alpha, float beta) {
    // Shared memory for sub-matrices
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // Thread indices
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Calculate the row and column of the block in the grid
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    // Calculate the position of the current thread's element in matrices A, B, and C
    float tmp = 0.0;

    // Advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    // Loop over sub-matrices of A and B
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Load data into shared memory
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + bkIdx + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[(bkIdx + threadRow) * N + threadCol];

        // Synchronize to ensure all threads have finished loading data
        __syncthreads();

        // Perform the dot product
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        // Synchronize to ensure all threads have finished computation before loading new data
        __syncthreads();

        // Move pointers to the next chunk
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    // Store the result into global memory
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

int main() {
    // Define matrix dimensions
    int M = 512; // Number of rows in A and C
    int K = 512; // Number of columns in A and rows in B
    int N = 512; // Number of columns in B and C

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_result = new float[M * N];

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);

    // Launch kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N, 1.0f, 0.0f);

    // Copy result back from device to host
    cudaMemcpy(h_C_result, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_result;

    return 0;
}

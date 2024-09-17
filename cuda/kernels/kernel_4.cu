#include <cuda_runtime.h>
#include <iostream>

#define BLOCKSIZE 8
#define TM 8       // Number of results each thread computes (tiling factor)
#define BK BLOCKSIZE

// Kernel function to perform matrix multiplication using shared memory
__global__ void matmul1D_blocktiling(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    // Allocate shared memory
    __shared__ float As[BLOCKSIZE * BK];
    __shared__ float Bs[BLOCKSIZE * BK];

    // Thread indices within the block
    int threadRow = threadIdx.y; // Local row in the block
    int threadCol = threadIdx.x; // Local column in the block

    // Block indices (row and column) in the grid
    int cRow = blockIdx.y; // Block row index
    int cCol = blockIdx.x; // Block column index

    // Allocate registers to store results (each thread computes TM results)
    float threadResults[TM] = {0.0f};

    // Initialize the pointer offsets for A, B, and C
    float *A_start = A + cRow * BLOCKSIZE * K;              // Row start in A
    float *B_start = B + cCol * BLOCKSIZE;                  // Column start in B
    float *C_start = C + cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // C start position

    // Outer loop over block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load a block of A and B into shared memory
        As[threadRow * BK + threadCol] = A_start[threadRow * K + threadCol + bkIdx];
        Bs[threadRow * BK + threadCol] = B_start[threadRow * N + threadCol + bkIdx * N];
        __syncthreads();

        // Compute the dot product for the current block tile
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Cache the value of B for reuse in the inner loop
            float Btmp = Bs[dotIdx * BLOCKSIZE + threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();
    }

    // Store the final results back to global memory
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        int globalRow = cRow * BLOCKSIZE + threadRow * TM + resIdx;
        int globalCol = cCol * BLOCKSIZE + threadCol;
        C_start[globalRow * N + globalCol] = alpha * threadResults[resIdx] + beta * C_start[globalRow * N + globalCol];
    }
}

// Host function to call the kernel
void matmul1D_blocktiling_host(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    // Define grid and block dimensions
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid(N / BLOCKSIZE, M / BLOCKSIZE);

    // Launch the kernel
    matmul1D_blocktiling<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);

    // Synchronize
    cudaDeviceSynchronize();
}

// Example usage
int main() {
    int M = 64, N = 64, K = 64;
    float alpha = 1.0f, beta = 0.0f;

    // Allocate and initialize matrices A, B, and C
    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));

    // Initialize A, B, and C with some values
    for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(i % 100);
    for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(i % 100);
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    // Call the matrix multiplication function
    matmul1D_blocktiling_host(A, B, C, M, N, K, alpha, beta);

    // Free the memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

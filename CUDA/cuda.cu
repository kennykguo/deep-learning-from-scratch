#include <cuda_runtime.h>
#include <iostream>

__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 100;
    int size = n * sizeof(int);
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    // Allocate memory on the host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Initialize arrays a and b
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy arrays a and b to the device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch the kernel on the device
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result array c back to the host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Error at index " << i << std::endl;
            break;
        }
    }

    // Free memory on the device and host
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    std::cout << "Completed successfully!" << std::endl;
    return 0;
}

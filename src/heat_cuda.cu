#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "initialize.h"

//--------
//
// Commands to learn on lab computers to fix cuda path.
//
// $ export PATH=/usr/local/cuda/bin:$PATH
// $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
// $ source ~/.bashrc
//
//--------

__global__ void update(double* U, double* U_next, int nx, int ny, double lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        U_next[i * ny + j] = (1 - 4 * lambda) * U[i * ny + j] 
                           + lambda * (U[(i+1) * ny + j] + U[i * ny + (j+1)] + U[(i-1) * ny + j] + U[i * ny + (j-1)]);
    }
}

void write_to_file(double* U, int nx, int ny, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            fprintf(fp, "%f ", U[i * ny + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    int size = NX * NY * sizeof(double);
    double *U = (double *)malloc(size);
    double *U_next = (double *)malloc(size);
    double *d_U, *d_U_next;

    checkCudaError(cudaMalloc((void **)&d_U, size), "Failed to allocate device memory for U");
    checkCudaError(cudaMalloc((void **)&d_U_next, size), "Failed to allocate device memory for U_next");

    initialize(U, NX, NY);

    checkCudaError(cudaMemcpy(d_U, U, size, cudaMemcpyHostToDevice), "Failed to copy U to device");
    checkCudaError(cudaMemcpy(d_U_next, U, size, cudaMemcpyHostToDevice), "Failed to copy U_next to device");

    double lambda = GAMMA / (DELTA * DELTA);
    if (lambda >= 0.5) {
        printf("Error: lambda = %f is not stable\n", lambda);
        exit(1);
    }

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create output directory if it does not exist
    struct stat st = {0};
    if (stat("output_cuda", &st) == -1) {
        mkdir("output_cuda", 0700);
    }

    for (int step = 0; step < N_STEPS; step++) {
        update<<<numBlocks, threadsPerBlock>>>(d_U, d_U_next, NX, NY, lambda);
        checkCudaError(cudaGetLastError(), "Kernel launch failed");
        checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

        // Swap pointers
        double* temp = d_U;
        d_U = d_U_next;
        d_U_next = temp;

        checkCudaError(cudaMemcpy(U, d_U, size, cudaMemcpyDeviceToHost), "Failed to copy U from device to host");

        char filename[100];
        sprintf(filename, "output_cuda/output_%d.dat", step);
        write_to_file(U, NX, NY, filename);
    }

    cudaFree(d_U);
    cudaFree(d_U_next);
    free(U);
    free(U_next);

    return 0;
}

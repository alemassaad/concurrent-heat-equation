#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/stat.h>
#include <sys/types.h>

#define NX 1000
#define NY 1000
#define DELTA 0.01
#define GAMMA 0.00001
#define N_STEPS 600

__global__ void update(double* U, double* U_next, int nx, int ny, double lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        U_next[i * ny + j] = (1 - 4 * lambda) * U[i * ny + j] 
                           + lambda * (U[(i+1) * ny + j] + U[i * ny + (j+1)] + U[(i-1) * ny + j] + U[i * ny + (j-1)]);
    }
}

void initialize(double* U, int nx, int ny) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            U[i * ny + j] = 0.0;
        }
    }
    U[(nx/2) * ny + (ny/2)] = 100000.0;
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

int main() {
    int size = NX * NY * sizeof(double);
    double *U = (double *)malloc(size);
    double *U_next = (double *)malloc(size);
    double *d_U, *d_U_next;

    cudaMalloc((void **)&d_U, size);
    cudaMalloc((void **)&d_U_next, size);

    initialize(U, NX, NY);

    cudaMemcpy(d_U, U, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U_next, U, size, cudaMemcpyHostToDevice);

    double lambda = GAMMA / (DELTA * DELTA);
    if (lambda >= 0.5) {
        printf("Error: lambda = %f is not stable\n", lambda);
        exit(1);
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create output directory if it does not exist
    struct stat st = {0};
    if (stat("output", &st) == -1) {
        mkdir("output", 0700);
    }

    for (int step = 0; step < N_STEPS; step++) {
        update<<<numBlocks, threadsPerBlock>>>(d_U, d_U_next, NX, NY, lambda);
        cudaMemcpy(d_U, d_U_next, size, cudaMemcpyDeviceToDevice);

        char filename[100];
        sprintf(filename, "output/output_%d.dat", step);
        cudaMemcpy(U, d_U, size, cudaMemcpyDeviceToHost);
        write_to_file(U, NX, NY, filename);
    }

    cudaFree(d_U);
    cudaFree(d_U_next);
    free(U);
    free(U_next);

    return 0;
}

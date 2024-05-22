#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define NX 100
#define NY 100
#define DELTA 0.01
#define GAMMA 0.00001
#define N_STEPS 600
#define TPB 16 // Threads per block

__global__ void updateKernel(double *U, double *U_next, double lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= 1 && i < NX-1 && j >= 1 && j < NY-1) {
        U_next[i * NY + j] = (1 - 4 * lambda) * U[i * NY + j] 
                           + lambda * (U[(i+1) * NY + j] + U[(i-1) * NY + j] 
                           + U[i * NY + (j+1)] + U[i * NY + (j-1)]);
    }
}

void update(double *U, double *U_next) {
    double lambda = GAMMA / (DELTA * DELTA);
    double *d_U, *d_U_next;

    cudaMalloc(&d_U, NX * NY * sizeof(double));
    cudaMalloc(&d_U_next, NX * NY * sizeof(double));

    cudaMemcpy(d_U, U, NX * NY * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blocks((NX + TPB - 1) / TPB, (NY + TPB - 1) / TPB);
    dim3 threads(TPB, TPB);
    updateKernel<<<blocks, threads>>>(d_U, d_U_next, lambda);
    cudaDeviceSynchronize();  // Ensure all threads complete

    cudaMemcpy(U_next, d_U_next, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_U);
    cudaFree(d_U_next);
}

void initialize(double *U) {
    memset(U, 0, NX * NY * sizeof(double));
    U[(NX/2) * NY + (NY/2)] = 100000.0;  // Set initial heat source
}

void write_to_file(double *U, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file\n");
        return;
    }
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(fp, "%f ", U[i * NY + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main() {
    double *U = (double*)malloc(NX * NY * sizeof(double));
    double *U_next = (double*)malloc(NX * NY * sizeof(double));
    initialize(U);

    for (int step = 0; step < N_STEPS; step++) {
        update(U, U_next);
        memcpy(U, U_next, NX * NY * sizeof(double));
        char filename[100];
        sprintf(filename, "output_%d.dat", step);
        write_to_file(U, filename);
    }

    free(U);
    free(U_next);
    return 0;
}
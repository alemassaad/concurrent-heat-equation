#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime.h>

#define NX 100
#define NY 100
#define DELTA 0.01
#define GAMMA 0.00001
#define N_STEPS 600
#define TPB 16 // Threads per block
#define MAX_ITER 1000 // Maximum iterations for Gauss-Seidel

// CUDA kernel for the implicit update using the Gauss-Seidel method
__global__ void implicitUpdateKernel(double *U, double *U_next, double lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < NX-1 && j >= 1 && j < NY-1) {
        double old_value, new_value;
        // Initialize U_next at the current position
        new_value = U[i * NY + j];
        
        for (int iter = 0; iter < MAX_ITER; iter++) {
            old_value = new_value;
            new_value = (U[i * NY + j] 
                        + lambda * (U_next[(i+1) * NY + j] 
                        + U_next[(i-1) * NY + j] 
                        + U_next[i * NY + (j+1)] 
                        + U_next[i * NY + (j-1)])) / (1.0 + 4.0 * lambda);

            // Convergence criterion (simple tolerance check)
            if (fabs(new_value - old_value) < 1e-5) {
                break;
            }

            U_next[i * NY + j] = new_value;
        }
    }
}

// Function to perform the update by calling the CUDA kernel
void update(double *U, double *U_next) {
    double lambda = GAMMA / (DELTA * DELTA);
    double *d_U, *d_U_next;

    cudaMalloc(&d_U, NX * NY * sizeof(double));
    cudaMalloc(&d_U_next, NX * NY * sizeof(double));

    cudaMemcpy(d_U, U, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U_next, U, NX * NY * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blocks((NX + TPB - 1) / TPB, (NY + TPB - 1) / TPB);
    dim3 threads(TPB, TPB);
    implicitUpdateKernel<<<blocks, threads>>>(d_U, d_U_next, lambda);
    cudaDeviceSynchronize();

    cudaMemcpy(U_next, d_U_next, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_U);
    cudaFree(d_U_next);
}

// Initializes the grid with an initial condition
void initialize(double *U) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            U[i * NY + j] = 0.0;
        }
    }
    U[(NX/2) * NY + (NY/2)] = 100000.0;  // High initial condition at center
}

// Writes the grid to a file
void write_to_file(double *U, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open file for writing");
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

    initialize(U); // Initialize U with the starting conditions

    // Simulation loop
    for (int step = 0; step < N_STEPS; step++) {
        update(U, U_next); // Perform the update
        memcpy(U, U_next, NX * NY * sizeof(double)); // Copy U_next back to U for the next step
        char filename[256];
        sprintf(filename, "output_%d.dat", step); // Create filename for output
        write_to_file(U, filename); // Write current state to file
    }

    free(U);
    free(U_next);
    return 0;
}
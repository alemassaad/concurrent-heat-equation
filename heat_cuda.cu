#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <assert.h>
#include "initialize.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__device__ int getIndex(const int i, const int j, const int width) {
    return i * width + j;
}

__global__ void update(float* U, float* U_next, int nx, int ny, float lambda) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;

    // Shared memory allocation with halo cells
    __shared__ float s_U[BLOCK_SIZE_X + 2][BLOCK_SIZE_Y + 2];

    // Load data into shared memory, including halo cells
    if (i < nx && j < ny) {
        s_U[tx + 1][ty + 1] = U[getIndex(i, j, ny)];

        if (tx == 0 && i > 0) s_U[0][ty + 1] = U[getIndex(i - 1, j, ny)];
        if (tx == BLOCK_SIZE_X - 1 && i < nx - 1) s_U[tx + 2][ty + 1] = U[getIndex(i + 1, j, ny)];
        if (ty == 0 && j > 0) s_U[tx + 1][0] = U[getIndex(i, j - 1, ny)];
        if (ty == BLOCK_SIZE_Y - 1 && j < ny - 1) s_U[tx + 1][ty + 2] = U[getIndex(i, j + 1, ny)];

        if (tx == 0 && ty == 0 && i > 0 && j > 0) s_U[0][0] = U[getIndex(i - 1, j - 1, ny)];
        if (tx == 0 && ty == BLOCK_SIZE_Y - 1 && i > 0 && j < ny - 1) s_U[0][ty + 2] = U[getIndex(i - 1, j + 1, ny)];
        if (tx == BLOCK_SIZE_X - 1 && ty == 0 && i < nx - 1 && j > 0) s_U[tx + 2][0] = U[getIndex(i + 1, j - 1, ny)];
        if (tx == BLOCK_SIZE_X - 1 && ty == BLOCK_SIZE_Y - 1 && i < nx - 1 && j < ny - 1) s_U[tx + 2][ty + 2] = U[getIndex(i + 1, j + 1, ny)];

        __syncthreads();

        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            float uij = s_U[tx + 1][ty + 1];
            float uim1j = s_U[tx][ty + 1];
            float uip1j = s_U[tx + 2][ty + 1];
            float uijm1 = s_U[tx + 1][ty];
            float uijp1 = s_U[tx + 1][ty + 2];

            float term = (1 - 4 * lambda) * uij + lambda * (uim1j + uip1j + uijm1 + uijp1);
            U_next[getIndex(i, j, ny)] = isnan(term) ? uij : term;
        }
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void test_filename_truncation() {
    char filename[256];
    int needed = snprintf(filename, sizeof(filename), "output_seq/very_long_filename_prefix_%d.dat", INT_MAX);
    assert(needed < sizeof(filename));
    printf("Test passed: Filename is within buffer limits.\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <run_num>\n", argv[0]);
        return 1;
    }

    int run_num = atoi(argv[1]);

    int size = NX * NY * sizeof(float);
    float* U = (float*)calloc(NX * NY, sizeof(float));
    float* U_next = (float*)calloc(NX * NY, sizeof(float));
    float* d_U, * d_U_next;

    checkCudaError(cudaMalloc((void**)&d_U, size), "Failed to allocate device memory for U");
    checkCudaError(cudaMalloc((void**)&d_U_next, size), "Failed to allocate device memory for U_next");

    initialize(U, NX, NY);

    for (int i = 0; i < NX * NY; i++) {
        if (isnan(U[i])) {
            printf("Initialization produced NaN at index %d\n", i);
            free(U);
            free(U_next);
            return 1;
        }
    }

    checkCudaError(cudaMemcpy(d_U, U, size, cudaMemcpyHostToDevice), "Failed to copy U to device");
    checkCudaError(cudaMemcpy(d_U_next, U, size, cudaMemcpyHostToDevice), "Failed to copy U_next to device");

    float lambda = GAMMA / (DELTA * DELTA);
    if (lambda >= 0.5) {
        printf("Warning: lambda = %f is approaching instability, adjusting...\n", lambda);
        lambda = 0.49; // Adjust lambda to maintain stability
    }
    
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create output directory if it does not exist
    struct stat st = {0};
    if (stat("output_cuda", &st) == -1) {
        mkdir("output_cuda", 0700);
        printf("Created directory: output_cuda\n");
    }
    if (stat("frames_cuda", &st) == -1) {
        mkdir("frames_cuda", 0700);
        printf("Created directory: frames_cuda\n");
    }

    char csv_filename[256];
    snprintf(csv_filename, sizeof(csv_filename), "results/results_%dx%d.csv", NX, NY);

    int last_complete_step = -1;
    FILE *csv_file = fopen(csv_filename, "r");
    if (csv_file != NULL) {
        char line[256];
        while (fgets(line, sizeof(line), csv_file)) {
            int step;
            char *ptr = strrchr(line, ',');
            if (ptr && *(ptr + 1) == '\n') {
                sscanf(line, "%*d,%*[^,],%*d,%d,%*f", &step);
                last_complete_step = step;
            }
        }
        fclose(csv_file);
    }

    csv_file = fopen(csv_filename, "a");
    if (csv_file == NULL) {
        fprintf(stderr, "Error opening CSV file %s\n", csv_filename);
        return 1;
    }

    struct timespec start, end;
    double cumulative_time = 0.0;

    for (int step = last_complete_step + 1; step <= N_STEPS; step++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        update<<<numBlocks, threadsPerBlock>>>(d_U, d_U_next, NX, NY, lambda);
        checkCudaError(cudaGetLastError(), "Kernel launch failed");
        checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Swap pointers
        float* temp = d_U;
        d_U = d_U_next;
        d_U_next = temp;

        double elapsed_time;
        if (end.tv_nsec < start.tv_nsec) {
            elapsed_time = (end.tv_sec - start.tv_sec - 1) + (end.tv_nsec + 1e9 - start.tv_nsec) * 1e-9;
        } else {
            elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
        }

        cumulative_time += elapsed_time;

        if (step % STEP_INTERVAL == 0) {
            checkCudaError(cudaMemcpy(U, d_U, size, cudaMemcpyDeviceToHost), "Failed to copy U from device to host");

            char filename[100];
            sprintf(filename, "output_cuda/output_%d.dat", step);
            FILE* fp = fopen(filename, "w");
            if (fp == NULL) {
                printf("Error opening file %s\n", filename);
                exit(1);
            }
            for (int i = 0; i < NX; i++) {
                for (int j = 0; j < NY; j++) {
                    fprintf(fp, "%f ", U[i * NY + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);

            fprintf(csv_file, "%d,cuda,%d,%d,%f\n", NX, run_num, step, cumulative_time);
            printf("Done step: %d in %f seconds (cumulative time: %f)\n", step, elapsed_time, cumulative_time);
        }
    }

    cudaFree(d_U);
    cudaFree(d_U_next);
    free(U);
    free(U_next);

    fclose(csv_file);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include "initialize.h"

void update(float *U, float *U_next, int nx, int ny);
void write_to_file(float *U, int nx, int ny, const char *filename);

int main() {
    clock_t start_time = clock();
    int nx = NX, ny = NY;
    float *U = (float *)malloc(nx * ny * sizeof(float));
    float *U_next = (float *)malloc(nx * ny * sizeof(float));

    if (U == NULL || U_next == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    initialize(U, nx, ny);

    struct stat st = {0};
    if (stat("output_seq", &st) == -1) {
        mkdir("output_seq", 0700);
    }
    if (stat("heatmaps_seq", &st) == -1) {
        mkdir("heatmaps_seq", 0700);
    }

    for (int step = 0; step < N_STEPS; step++) {
        update(U, U_next, nx, ny);
        memcpy(U, U_next, nx * ny * sizeof(float));
        if (step % STEP_INTERVAL == 0) {  
            char filename[100];
            sprintf(filename, "output_seq/output_%d.dat", step);
            write_to_file(U, nx, ny, filename);
        }
    }

    // Ensuring the last step is always saved
    char filename[100];
    sprintf(filename, "output_seq/output_%d.dat", N_STEPS);
    write_to_file(U, nx, ny, filename);

    free(U);
    free(U_next);

    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Sequential execution time: %f seconds\n", time_spent);


    return 0;
}

void update(float *U, float *U_next, int nx, int ny) {
    float lambda = GAMMA / (DELTA * DELTA);

    if (lambda >= 0.5) {
        printf("Error: lambda = %f is not stable\n", lambda);
        exit(1);
    }

    for (int i = 1; i < nx-1; i++) {
        for (int j = 1; j < ny-1; j++) {
            float term = (1 - 4 * lambda) * U[i * ny + j] 
                        + lambda * (U[(i+1) * ny + j] + U[i * ny + (j+1)] + U[(i-1) * ny + j] + U[i * ny + (j-1)]);

            if (isnan(term)) {
                printf("NaN detected at i=%d, j=%d\n", i, j);
                exit(1);
            }
            U_next[i * ny + j] = term;
        }
    }

    for (int i = 0; i < nx; i++) {
        U_next[i * ny] = U_next[i * ny + 1];
        U_next[i * ny + (ny-1)] = U_next[i * ny + (ny-2)];
    }
    for (int j = 0; j < ny; j++) {
        U_next[j] = U_next[ny + j];
        U_next[(nx-1) * ny + j] = U_next[(nx-2) * ny + j];
    }
}

void write_to_file(float *U, int nx, int ny, const char *filename) {
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

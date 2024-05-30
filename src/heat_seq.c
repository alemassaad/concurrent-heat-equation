#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "initialize.h"

#define NX 100
#define NY 100
#define DELTA 0.01
#define GAMMA 0.00001  // Adjusted for stability
#define N_STEPS 600

void update(double *U, double *U_next, int nx, int ny);
void write_to_file(double *U, int nx, int ny, const char *filename);

int main() {
    int nx = NX, ny = NY;
    double *U = (double *)malloc(nx * ny * sizeof(double));
    double *U_next = (double *)malloc(nx * ny * sizeof(double));

    if (U == NULL || U_next == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    initialize(U, nx, ny);

    for (int step = 0; step < N_STEPS; step++) {
        update(U, U_next, nx, ny);
        memcpy(U, U_next, nx * ny * sizeof(double));
        char filename[100];
        sprintf(filename, "output_seq/output_%d.dat", step);
        write_to_file(U, nx, ny, filename);
    }

    free(U);
    free(U_next);

    return 0;
}

void update(double *U, double *U_next, int nx, int ny) {
    double lambda = GAMMA / (DELTA * DELTA);

    if (lambda >= 0.5) {
        printf("Error: lambda = %f is not stable\n", lambda);
        exit(1);
    }

    for (int i = 1; i < nx-1; i++) {
        for (int j = 1; j < ny-1; j++) {
            double term = (1 - 4 * lambda) * U[i * ny + j] 
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

void write_to_file(double *U, int nx, int ny, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file\n");
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

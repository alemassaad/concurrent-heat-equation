#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NX 100
#define NY 100
#define DELTA 0.01
#define GAMMA 0.00001  // Adjusted for stability
#define N_STEPS 600

void initialize(double U[NX][NY]);
void update(double U[NX][NY], double U_next[NX][NY]);
void write_to_file(double U[NX][NY], const char *filename);

int main() {
    double U[NX][NY], U_next[NX][NY];
    initialize(U);

    for (int step = 0; step < N_STEPS; step++) {
        update(U, U_next);
        memcpy(U, U_next, sizeof(U));
        char filename[100];
        sprintf(filename, "output/output_%d.dat", step);
        write_to_file(U, filename);
    }

    return 0;
}

void initialize(double U[NX][NY]) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            U[i][j] = 0.0;
        }
    }
    U[NX/2][NY/2] = 100000.0;  // Increase initial heat source
}

void update(double U[NX][NY], double U_next[NX][NY]) {
    double lambda = GAMMA / (DELTA * DELTA);

    if (lambda >= 0.5) {
        printf("Error: lambda = %f is not stable\n", lambda);
        exit(1);
    }

    for (int i = 1; i < NX-1; i++) {
        for (int j = 1; j < NY-1; j++) {
            double term = (1 - 4 * lambda) * U[i][j] 
                        + lambda * (U[i+1][j] + U[i][j+1] + U[i-1][j] + U[i][j-1]);

            if (isnan(term)) {
                printf("NaN detected at i=%d, j=%d\n", i, j);
                exit(1);
            }
            U_next[i][j] = term;
        }
    }

    for (int i = 0; i < NX; i++) {
        U_next[i][0] = U_next[i][1];
        U_next[i][NY-1] = U_next[i][NY-2];
    }
    for (int j = 0; j < NY; j++) {
        U_next[0][j] = U_next[1][j];
        U_next[NX-1][j] = U_next[NX-2][j];
    }
}

void write_to_file(double U[NX][NY], const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file\n");
        exit(1);
    }
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(fp, "%f ", U[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

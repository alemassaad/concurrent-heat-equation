#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // Include for memcpy

#define NX 100  // Number of x grid points
#define NY 100  // Number of y grid points
#define DX 0.01 // Grid spacing in x direction
#define DY 0.01 // Grid spacing in y direction
#define DT 0.01 // Time step
#define N_STEPS 1000 // Number of time steps
#define KAPPA 0.1  // Diffusion coefficient

void initialize(double u[NX][NY]);
void update(double u[NX][NY], double u_next[NX][NY], int step);
void write_to_file(double u[NX][NY]);

int main() {
    double u[NX][NY], u_next[NX][NY];
    initialize(u);
    for (int step = 0; step < N_STEPS; step++) {
        update(u, u_next, step);  // Pass step here
        memcpy(u, u_next, sizeof(u));
        if (step % 100 == 0) {
            write_to_file(u);
        }
    }
    return 0;
}

void initialize(double u[NX][NY]) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            u[i][j] = 0.0;
        }
    }
    u[NX/2][NY/2] = 100.0; // Initial heat source at the center
}

void update(double u[NX][NY], double u_next[NX][NY], int step) {
    for (int i = 1; i < NX-1; i++) {
        for (int j = 1; j < NY-1; j++) {
            double u_xx = (u[i+1][j] - 2*u[i][j] + u[i-1][j]) / (DX*DX);
            double u_yy = (u[i][j+1] - 2*u[i][j] + u[i][j-1]) / (DY*DY);
            u_next[i][j] = u[i][j] + DT * KAPPA * (u_xx + u_yy);
            if (isnan(u_next[i][j])) {
                printf("NaN detected at step %d, i=%d, j=%d\n", step, i, j);
                printf("u_xx: %f, u_yy: %f, u[i][j]: %f\n", u_xx, u_yy, u[i][j]);
            }
        }
    }
}


void write_to_file(double u[NX][NY]) {
    FILE *fp = fopen("output.dat", "w");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(fp, "%f ", u[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

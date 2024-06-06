#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <time.h>
#include "initialize.h"

#define OWNER_FULL_PERMISSIONS 0700 // Permissions: owner read, write, execute only

void update(float *U, float *U_next, int nx, int ny);
void write_to_file(const float *U, int nx, int ny, const char *filename);
void create_directory(const char *dirname);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <run_num>\n", argv[0]);
        return 1;
    }

    int run_num = atoi(argv[1]);

    int nx = NX, ny = NY;
    float *U = (float *)malloc(nx * ny * sizeof(float));
    float *U_next = (float *)malloc(nx * ny * sizeof(float));

    if (U == NULL || U_next == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    initialize(U, nx, ny);
    create_directory("output_seq");
    create_directory("frames_seq");

    float lambda = GAMMA / (DELTA * DELTA);
    if (lambda >= 0.5) {
        fprintf(stderr, "Error: lambda = %f is not stable\n", lambda);
        free(U);
        free(U_next);
        return 1;
    }

    char csv_filename[256];
    snprintf(csv_filename, sizeof(csv_filename), "results/results_%dx%d.csv", nx, ny);

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
        update(U, U_next, nx, ny);
        clock_gettime(CLOCK_MONOTONIC, &end);

        float *temp = U;
        U = U_next;
        U_next = temp;

        double elapsed_time;
        if (end.tv_nsec < start.tv_nsec) {
            elapsed_time = (end.tv_sec - start.tv_sec - 1) + (end.tv_nsec + 1e9 - start.tv_nsec) * 1e-9;
        } else {
            elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
        }

        cumulative_time += elapsed_time;

        if (step % STEP_INTERVAL == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "output_seq/output_%d.dat", step);
            write_to_file(U, nx, ny, filename);
            fprintf(csv_file, "%d,seq,%d,%d,%f\n", nx, run_num, step, cumulative_time);
            printf("Done step %d in %f seconds (cumulative time: %f)\n", step, elapsed_time, cumulative_time);
        }
    }

    // Ensuring the last step is always saved
    char filename[256];
    snprintf(filename, sizeof(filename), "output_seq/output_%d.dat", N_STEPS);
    write_to_file(U, nx, ny, filename);

    fclose(csv_file);
    free(U);
    free(U_next);

    return 0;
}

void update(float *U, float *U_next, int nx, int ny) {
    float lambda = GAMMA / (DELTA * DELTA);

    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            float term = (1 - 4 * lambda) * U[i * ny + j] 
                        + lambda * (U[(i + 1) * ny + j] + U[i * ny + (j + 1)] + U[(i - 1) * ny + j] + U[i * ny + (j - 1)]);

            if (isnan(term)) {
                fprintf(stderr, "NaN detected at i=%d, j=%d\n", i, j);
                exit(1);
            }
            U_next[i * ny + j] = term;
        }
    }

    // Apply boundary conditions
    for (int i = 0; i < nx; i++) {
        U_next[i * ny] = U_next[i * ny + 1];
        U_next[i * ny + (ny - 1)] = U_next[i * ny + (ny - 2)];
    }
    for (int j = 0; j < ny; j++) {
        U_next[j] = U_next[ny + j];
        U_next[(nx - 1) * ny + j] = U_next[(nx - 2) * ny + j];
    }
}

void write_to_file(const float *U, int nx, int ny, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
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

void create_directory(const char *dirname) {
    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        if (mkdir(dirname, OWNER_FULL_PERMISSIONS) != 0) {
            fprintf(stderr, "Error creating directory %s\n", dirname);
            exit(1);
        }
    }
}

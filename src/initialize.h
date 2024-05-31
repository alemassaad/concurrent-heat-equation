// initialize.h

#ifndef INITIALIZE_H
#define INITIALIZE_H

#define NX 100
#define NY 100
#define DELTA 0.01
#define GAMMA 0.00001
#define N_STEPS 600

typedef struct {
    int x;
    int y;
    double intensity;
} HeatSource;

void initialize(double* U, int nx, int ny) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            U[i * ny + j] = 0.0;
        }
    }

    HeatSource sources[] = {
        {nx / 2, ny / 2, 100000.0}
    };
    int num_sources = sizeof(sources) / sizeof(HeatSource);

    for (int s = 0; s < num_sources; s++) {
        int x = sources[s].x;
        int y = sources[s].y;
        double intensity = sources[s].intensity;
        if (x >= 0 && x < nx && y >= 0 && y < ny) {
            U[x * ny + y] = intensity;
        }
    }
}

#endif

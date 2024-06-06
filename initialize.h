#ifndef INITIALIZE_H
#define INITIALIZE_H

#define NX 800
#define NY 800
#define DELTA 0.01f
#define GAMMA 0.00001f
#define N_STEPS 4000
#define STEP_INTERVAL 200
#define HEAT_INTENSITY 7500000.0f

typedef struct {
    size_t x;
    size_t y;
    float intensity;
} HeatSource;

void initialize(float* U, size_t nx, size_t ny) {
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            U[i * ny + j] = 0.0f;
        }
    }

    HeatSource sources[] = {
        {nx / 2, ny / 2, HEAT_INTENSITY}
    };
    size_t num_sources = sizeof(sources) / sizeof(HeatSource);

    for (size_t s = 0; s < num_sources; s++) {
        size_t x = sources[s].x;
        size_t y = sources[s].y;
        float intensity = sources[s].intensity;
        if (x < nx && y < ny) {
            U[x * ny + y] = intensity;
        }
    }
}

#endif

#ifndef INITIALIZE_H
#define INITIALIZE_H

#define NX 800
#define NY 800
#define DELTA 0.01f
#define GAMMA 0.00001f
#define N_STEPS 1000
#define STEP_INTERVAL 100
#define HEAT_INTENSITY 7500000.0f

typedef struct {
    size_t x;
    size_t y;
    float intensity;
} HeatSource;

void apply_heat_sources(float* U, HeatSource* sources, size_t num_sources, size_t nx, size_t ny) {
    for (size_t s = 0; s < num_sources; s++) {
        size_t x = sources[s].x;
        size_t y = sources[s].y;
        float intensity = sources[s].intensity;
        if (x < nx && y < ny) {
            U[x * ny + y] = intensity;
        }
    }
}

void initialize_single_heat_source(float* U, size_t nx, size_t ny) {
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            U[i * ny + j] = 0.0f;
        }
    }

    HeatSource source = {nx / 2, ny / 2, HEAT_INTENSITY};
    apply_heat_sources(U, &source, 1, nx, ny);
}

void initialize_multiple_heat_sources(float* U, size_t nx, size_t ny) {
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            U[i * ny + j] = 0.0f;
        }
    }

    HeatSource sources[] = {
        {nx / 2, ny / 2, HEAT_INTENSITY}, // Center
        {nx / 3, ny / 3, HEAT_INTENSITY / 2}, // Top-left quadrant
        {2 * nx / 3, 2 * ny / 3, HEAT_INTENSITY / 2}, // Bottom-right quadrant
        {nx / 3, 2 * ny / 3, -HEAT_INTENSITY / 2}, // Bottom-left quadrant
        {2 * nx / 3, ny / 3, -HEAT_INTENSITY / 2}, // Top-right quadrant
        {nx / 2, ny / 4, HEAT_INTENSITY / 3}, // Top-center
        {nx / 2, 3 * ny / 4, -HEAT_INTENSITY / 3}, // Bottom-center
        {nx / 4, ny / 2, HEAT_INTENSITY / 4}, // Left-center
        {3 * nx / 4, ny / 2, -HEAT_INTENSITY / 4}, // Right-center
        {nx / 2 + nx / 6, ny / 2 + ny / 6, HEAT_INTENSITY / 5}, // Slightly off-center positive
        {nx / 2 - nx / 6, ny / 2 - ny / 6, -HEAT_INTENSITY / 5} // Slightly off-center negative
    };
    size_t num_sources = sizeof(sources) / sizeof(HeatSource);

    apply_heat_sources(U, sources, num_sources, nx, ny);
}


void initialize(float* U, size_t nx, size_t ny) {
    // Comment/uncomment the initialization method you want to use:

    // Single heat source
    // initialize_single_heat_source(U, nx, ny);

    // Multiple heat sources
    initialize_multiple_heat_sources(U, nx, ny);
}

#endif

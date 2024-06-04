# Heat Equation Solver in CUDA
## Overview
This project implements a solver for the heat equation on a 2D plane (https://en.wikipedia.org/wiki/Heat_equation) using CUDA for efficient parallel processing on GPUs. The solver is designed to compare the performance of sequential and CUDA implementations.
## Requirements
- CUDA Toolkit: Ensure the CUDA toolkit is installed.
- GNUplot: Required for plotting the heatmaps.
- ImageMagick: Required for creating GIFs.
## Setup Instructions
Environment Setup:
    $ export PATH=/usr/local/cuda/bin:$PATH
    $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    $ source ~/.bashrc
## How to run
Run sequential and CUDA once:
    $ make 
Run sequential and CUDA systematically for a list of parameters:
    $ make performance
Clean up:
    $ make clean
## Customization
Modify parameters in initialize.h to customize the simulation:
    #define NX 1200
    #define NY 1200
    #define DELTA 0.01f
    #define GAMMA 0.00001f
    #define N_STEPS 300
    #define STEP_INTERVAL 100
    #define HEAT_INTENSITY 100000.0f
To systematically test multiple configurations, edit compare_performance.sh:
    GRID_SIZES=(1000 1200)
    NUM_STEPS=($(seq 100 100 300))
    RUNS=2
    STEP_INTERVAL=100
## Conclusion
This project demonstrates the performance benefits of using CUDA for parallelizing computational tasks like solving the heat equation. By following the instructions, you can execute, compare, and visualize the performance of sequential and parallel implementations.
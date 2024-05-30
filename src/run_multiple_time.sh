#!/bin/bash

# Number of times to run the program
NUM_RUNS=100

# Compiling the CUDA program
nvcc -o heat_simulation heat_equation.cu

# Loop to run the program multiple times
for ((i=1; i<=NUM_RUNS; i++))
do
    echo "Run $i"
    ./heat_simulation
done




#How to run:
#   chmod +x run_multiple_times.sh
#   ./run_multiple_times.sh

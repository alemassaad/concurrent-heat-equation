#!/bin/bash

# Define grid sizes and number of steps
GRID_SIZES=(100 200 300)
NUM_STEPS=($(seq 100 100 20000))

# Create a results directory
mkdir -p results

# Function to run the sequential version and measure time
run_seq() {
    NX=$1
    NY=$1
    STEPS=$2
    gcc -o heat_seq heat_seq.c -lm
    export NX
    export NY
    export N_STEPS=$STEPS
    { time -p ./heat_seq > /dev/null; } 2> temp_seq.txt
    grep 'real' temp_seq.txt | awk '{print $2}' > results/seq_${NX}_${STEPS}.txt
    rm -f heat_seq temp_seq.txt
}

# Function to run the CUDA version and measure time
run_cuda() {
    NX=$1
    NY=$1
    STEPS=$2
    nvcc -o heat_parallel heat_parallel.cu
    export NX
    export NY
    export N_STEPS=$STEPS
    { time -p ./heat_parallel > /dev/null; } 2> temp_cuda.txt
    grep 'real' temp_cuda.txt | awk '{print $2}' > results/cuda_${NX}_${STEPS}.txt
    rm -f heat_parallel temp_cuda.txt
}

# Run tests for different grid sizes and number of steps
for GRID_SIZE in "${GRID_SIZES[@]}"; do
    for STEPS in "${NUM_STEPS[@]}"; do
        echo "Running sequential version with grid size ${GRID_SIZE}x${GRID_SIZE} and ${STEPS} steps..."
        run_seq $GRID_SIZE $STEPS
        echo "Running CUDA version with grid size ${GRID_SIZE}x${GRID_SIZE} and ${STEPS} steps..."
        run_cuda $GRID_SIZE $STEPS
    done
done

echo "Performance tests completed. Generating graphs..."

# Create Python script to generate graphs
cat <<EOF > generate_graph.py
import matplotlib.pyplot as plt
import os

# Define grid sizes and number of steps
grid_sizes = [100, 200, 300]
num_steps = list(range(100, 2100, 100))

seq_times = []
cuda_times = []

def convert_time_to_seconds(time_str):
    return float(time_str.strip())

# Parse results
for grid_size in grid_sizes:
    for steps in num_steps:
        seq_file = f"results/seq_{grid_size}_{steps}.txt"
        cuda_file = f"results/cuda_{grid_size}_{steps}.txt"

        with open(seq_file, 'r') as f:
            lines = f.readlines()
            if lines:
                seq_time = lines[0].strip()
                seq_times.append((grid_size, steps, convert_time_to_seconds(seq_time)))
            else:
                seq_times.append((grid_size, steps, float('inf')))  # Handle missing data

        with open(cuda_file, 'r') as f:
            lines = f.readlines()
            if lines:
                cuda_time = lines[0].strip()
                cuda_times.append((grid_size, steps, convert_time_to_seconds(cuda_time)))
            else:
                cuda_times.append((grid_size, steps, float('inf')))  # Handle missing data

# Plot results
for grid_size in grid_sizes:
    seq_data = [t[2] for t in seq_times if t[0] == grid_size]
    cuda_data = [t[2] for t in cuda_times if t[0] == grid_size]

    plt.figure(figsize=(10, 6))
    plt.plot(num_steps, seq_data, label=f'Seq {grid_size}x{grid_size}', marker='o')
    plt.plot(num_steps, cuda_data, label=f'CUDA {grid_size}x{grid_size}', marker='x')
    plt.xlabel('Number of Steps')
    plt.ylabel('Time (s)')
    plt.title(f'Performance Comparison: Sequential vs CUDA ({grid_size}x{grid_size})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'performance_comparison_{grid_size}.png')
    plt.close()

EOF

# Run Python script to generate graphs
python3 generate_graph.py

echo "Graphs generated: performance_comparison_*.png"

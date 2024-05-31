#!/bin/bash

# Define main variables
GRID_SIZES=(100 300)
NUM_STEPS=($(seq 100 200 2000))
RUNS=10

# Create a results directory
mkdir -p results

# Function to get the elapsed time
print_elapsed_time() {
    local start_time=$1
    local message=$2
    local elapsed_time=$(($(date +%s) - start_time))
    local hours=$((elapsed_time / 3600))
    local minutes=$(( (elapsed_time % 3600) / 60 ))
    local seconds=$((elapsed_time % 60))
    printf "%02d:%02d:%02d: %s...\n" $hours $minutes $seconds "$message"
}

# Function to run the sequential version and measure time
run_seq() {
    NX=$1
    NY=$1
    STEPS=$2
    gcc -o heat_seq heat_seq.c -lm
    export NX
    export NY
    export N_STEPS=$STEPS
    TOTAL_TIME=0
    for ((i=1; i<=RUNS; i++)); do
        { time -p ./heat_seq > /dev/null; } 2> temp_seq.txt
        RUN_TIME=$(grep 'real' temp_seq.txt | awk '{print $2}')
        TOTAL_TIME=$(echo "$TOTAL_TIME + $RUN_TIME" | bc)
    done
    AVG_TIME=$(echo "scale=2; $TOTAL_TIME / $RUNS" | bc)
    echo $AVG_TIME > results/seq_${NX}_${STEPS}.txt
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
    TOTAL_TIME=0
    for ((i=1; i<=RUNS; i++)); do
        { time -p ./heat_parallel > /dev/null; } 2> temp_cuda.txt
        RUN_TIME=$(grep 'real' temp_cuda.txt | awk '{print $2}')
        TOTAL_TIME=$(echo "$TOTAL_TIME + $RUN_TIME" | bc)
    done
    AVG_TIME=$(echo "scale=2; $TOTAL_TIME / $RUNS" | bc)
    echo $AVG_TIME > results/cuda_${NX}_${STEPS}.txt
    rm -f heat_parallel temp_cuda.txt
}

# Start time
start_time=$(date +%s)

# Run tests for different grid sizes and number of steps
for GRID_SIZE in "${GRID_SIZES[@]}"; do
    for STEPS in "${NUM_STEPS[@]}"; do
        print_elapsed_time $start_time "Running sequential version with grid size ${GRID_SIZE}x${GRID_SIZE} and ${STEPS} steps"
        run_seq $GRID_SIZE $STEPS
        print_elapsed_time $start_time "Running CUDA version with grid size ${GRID_SIZE}x${GRID_SIZE} and ${STEPS} steps"
        run_cuda $GRID_SIZE $STEPS
    done

    print_elapsed_time $start_time "Generating graph for grid size ${GRID_SIZE}"

    # Create Python script to generate graph for the current grid size
    echo "Debug: Creating Python script for grid size ${GRID_SIZE}"
    cat <<EOF > generate_graph_${GRID_SIZE}.py
import matplotlib.pyplot as plt
import os

# Define grid size and number of steps
grid_size = ${GRID_SIZE}
num_steps = [$(IFS=,; echo "${NUM_STEPS[*]}")]

seq_times = []
cuda_times = []

def convert_time_to_seconds(time_str):
    return float(time_str.strip())

# Debug prints
print(f"Debug: num_steps = {num_steps}")
print(f"Debug: grid_size = {grid_size}")

# Parse results
for steps in num_steps:
    seq_file = f"results/seq_{grid_size}_{steps}.txt"
    cuda_file = f"results/cuda_{grid_size}_{steps}.txt"

    print(f"Debug: Checking files {seq_file} and {cuda_file}")

    try:
        with open(seq_file, 'r') as f:
            lines = f.readlines()
            if lines:
                seq_time = lines[0].strip()
                seq_times.append(convert_time_to_seconds(seq_time))
            else:
                seq_times.append(float('inf'))  # Handle missing data
    except FileNotFoundError:
        print(f"Debug: File not found {seq_file}")
        seq_times.append(float('inf'))

    try:
        with open(cuda_file, 'r') as f:
            lines = f.readlines()
            if lines:
                cuda_time = lines[0].strip()
                cuda_times.append(convert_time_to_seconds(cuda_time))
            else:
                cuda_times.append(float('inf'))  # Handle missing data
    except FileNotFoundError:
        print(f"Debug: File not found {cuda_file}")
        cuda_times.append(float('inf'))

print(f"Debug: seq_times = {seq_times}")
print(f"Debug: cuda_times = {cuda_times}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(num_steps, seq_times, label=f'Seq {grid_size}x{grid_size}', marker='o')
plt.plot(num_steps, cuda_times, label=f'CUDA {grid_size}x{grid_size}', marker='x')
plt.xlabel('Number of Steps')
plt.ylabel('Time (s)')
plt.title(f'Performance Comparison: Sequential vs CUDA ({grid_size}x{grid_size})')
plt.legend()
plt.grid(True)
output_filename = f'performance_comparison_{grid_size}.png'
print(f"Debug: Saving figure as {output_filename}")
plt.savefig(output_filename)
plt.close()

EOF

    # Debug print to check generated Python script
    echo "Debug: Content of generate_graph_${GRID_SIZE}.py"
    cat generate_graph_${GRID_SIZE}.py

    # Run Python script to generate graph
    python3 generate_graph_${GRID_SIZE}.py

    print_elapsed_time $start_time "Graph generated: performance_comparison_${GRID_SIZE}"
done

# Final elapsed time
print_elapsed_time $start_time "All performance tests and graph generation completed"

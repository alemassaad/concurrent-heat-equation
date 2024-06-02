#!/bin/bash

# Define main variables
GRID_SIZES=(100)
NUM_STEPS=($(seq 200 200 5000))
RUNS=2

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
    RUN_INDEX=$3
    gcc -o heat_seq heat_seq.c -lm
    export NX=$NX
    export NY=$NY
    export N_STEPS=$STEPS
    mkdir -p output_seq heatmaps_seq
    TOTAL_TIME=0
    { time -p ./heat_seq > /dev/null; } 2> temp_seq.txt
    RUN_TIME=$(grep 'real' temp_seq.txt | awk '{print $2}')
    TOTAL_TIME=$(echo "$TOTAL_TIME + $RUN_TIME" | bc)
    echo $RUN_TIME > results/seq_${NX}_${STEPS}.txt
    rm -f temp_seq.txt

    # Generate Gnuplot script and heatmaps
    ./generate_gnuplot_script.sh seq $STEPS
    gnuplot plot_seq.gp

    # Generate GIFs
    GIF_DIR="gifs/seq/${NX}x${NY}"
    mkdir -p $GIF_DIR
    if ls heatmaps_seq/*.png 1> /dev/null 2>&1; then
        convert -delay 10 -loop 0 heatmaps_seq/*.png "${GIF_DIR}/heatmap_seq_${NX}x${NY}_${STEPS}steps_run${RUN_INDEX}.gif"
    else
        echo "No PNG files found for heatmaps_seq/*.png"
    fi
}

# Function to run the CUDA version and measure time
run_cuda() {
    NX=$1
    NY=$1
    STEPS=$2
    RUN_INDEX=$3
    nvcc -o heat_cuda heat_cuda.cu
    export NX=$NX
    export NY=$NY
    export N_STEPS=$STEPS
    mkdir -p output_cuda heatmaps_cuda
    TOTAL_TIME=0
    { time -p ./heat_cuda > /dev/null; } 2> temp_cuda.txt
    RUN_TIME=$(grep 'real' temp_cuda.txt | awk '{print $2}')
    TOTAL_TIME=$(echo "$TOTAL_TIME + $RUN_TIME" | bc)
    echo $RUN_TIME > results/cuda_${NX}_${STEPS}.txt
    rm -f temp_cuda.txt

    # Generate Gnuplot script and heatmaps
    ./generate_gnuplot_script.sh cuda $STEPS
    gnuplot plot_cuda.gp

    # Generate GIFs
    GIF_DIR="gifs/cuda/${NX}x${NY}"
    mkdir -p $GIF_DIR
    if ls heatmaps_cuda/*.png 1> /dev/null 2>&1; then
        convert -delay 10 -loop 0 heatmaps_cuda/*.png "${GIF_DIR}/heatmap_cuda_${NX}x${NY}_${STEPS}steps_run${RUN_INDEX}.gif"
    else
        echo "No PNG files found for heatmaps_cuda/*.png"
    fi
}

# Start time
start_time=$(date +%s)

# Run tests for different grid sizes and number of steps
for GRID_SIZE in "${GRID_SIZES[@]}"; do
    for STEPS in "${NUM_STEPS[@]}"; do
        for ((RUN_INDEX=1; RUN_INDEX<=RUNS; RUN_INDEX++)); do
            print_elapsed_time $start_time "Running sequential version with grid size ${GRID_SIZE}x${GRID_SIZE}, ${STEPS} steps, run ${RUN_INDEX}"
            run_seq $GRID_SIZE $STEPS $RUN_INDEX
            print_elapsed_time $start_time "Running CUDA version with grid size ${GRID_SIZE}x${GRID_SIZE}, ${STEPS} steps, run ${RUN_INDEX}"
            run_cuda $GRID_SIZE $STEPS $RUN_INDEX
        done
    done

    print_elapsed_time $start_time "Generating graph for grid size ${GRID_SIZE}"

    # Create Python script to generate graph for the current grid size
    cat <<EOF > generate_graph_${GRID_SIZE}.py
import matplotlib.pyplot as plt

# Define grid size and number of steps
grid_size = ${GRID_SIZE}
num_steps = [$(IFS=,; echo "${NUM_STEPS[*]}")]

seq_times = []
cuda_times = []

def convert_time_to_seconds(time_str):
    return float(time_str.strip())

# Parse results
for steps in num_steps:
    seq_file = f"results/seq_{grid_size}_{steps}.txt"
    cuda_file = f"results/cuda_{grid_size}_{steps}.txt"

    try:
        with open(seq_file, 'r') as f:
            lines = f.readlines()
            if lines:
                seq_time = lines[0].strip()
                seq_times.append(convert_time_to_seconds(seq_time))
            else:
                seq_times.append(float('inf'))  # Handle missing data
    except FileNotFoundError:
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
        cuda_times.append(float('inf'))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(num_steps, seq_times, label=f'Seq {grid_size}x{grid_size}', marker='o')
plt.plot(num_steps, cuda_times, label=f'CUDA {grid_size}x{grid_size}', marker='x')
plt.xlabel('Number of Steps')
plt.ylabel('Time (s)')
plt.title(f'Performance Comparison: Sequential vs CUDA ({grid_size}x{grid_size})')
plt.legend()
plt.grid(True)
plt.savefig(f'performance_comparison_{grid_size}.png')
plt.close()
EOF

    # Run Python script to generate graph
    python3 generate_graph_${GRID_SIZE}.py

    print_elapsed_time $start_time "Graph generated: performance_comparison_${GRID_SIZE}"
done

# Final elapsed time
print_elapsed_time $start_time "All performance tests and graph generation completed"

#!/bin/bash

# Define main variables
GRID_SIZES=(600 800)
NUM_STEPS=1000
RUNS=2
STEP_INTERVAL=100

# Function to create necessary directories
create_directories() {
    local grid_size=$1
    echo "Creating directories for grid size: $grid_size"  # Add this line
    mkdir -p results frames_seq frames_cuda gifs/seq/${grid_size}x${grid_size} gifs/cuda/${grid_size}x${grid_size}
}

# Function to modify initialize.h
modify_initialize() {
    local size=$1
    local steps=$2
    local interval=$3

    sed -i "s/^#define NX .*/#define NX $size/" initialize.h
    sed -i "s/^#define NY .*/#define NY $size/" initialize.h
    sed -i "s/^#define N_STEPS .*/#define N_STEPS $steps/" initialize.h
    sed -i "s/^#define STEP_INTERVAL .*/#define STEP_INTERVAL $interval/" initialize.h
}

# Function to create necessary directories
create_directories() {
    local grid_size=$1
    mkdir -p results frames_seq frames_cuda gifs/seq/${grid_size}x${grid_size} gifs/cuda/${grid_size}x${grid_size}
}

# Function to run the simulation and generate GIFs
run_simulation() {
    local grid_size=$1
    local run_num=$2
    local exec_type=$3

    if [[ $exec_type == "seq" ]]; then
        make -s compile_seq
        start_time=$(date +%s%N)
        make -s execute_seq run_num=$run_num
        end_time=$(date +%s%N)
        seq_time=$(echo "scale=9; ($end_time - $start_time) / 1000000000" | bc -l)
        seq_total_time=$(echo "$seq_total_time + $seq_time" | bc -l)
        echo "Run $run_num done in $seq_time seconds (seq)."

        echo "Generating GIF..."
        ./generate_gnuplot.sh seq $NUM_STEPS $STEP_INTERVAL $grid_size $grid_size
        GIF_PATH_SEQ=gifs/seq/${grid_size}x${grid_size}/heat_seq_${grid_size}x${grid_size}_${NUM_STEPS}steps_run${run_num}.gif
        convert -delay 10 -loop 0 $(ls frames_seq/*.png | sort -V) $GIF_PATH_SEQ
        echo "Generated $GIF_PATH_SEQ."

    elif [[ $exec_type == "cuda" ]]; then
        make -s compile_cuda
        start_time=$(date +%s%N)
        make -s execute_cuda run_num=$run_num
        end_time=$(date +%s%N)
        cuda_time=$(echo "scale=9; ($end_time - $start_time) / 1000000000" | bc -l)
        cuda_total_time=$(echo "$cuda_total_time + $cuda_time" | bc -l)
        echo "Run $run_num done in $cuda_time seconds (cuda)."

        echo "Generating GIF..."
        ./generate_gnuplot.sh cuda $NUM_STEPS $STEP_INTERVAL $grid_size $grid_size
        GIF_PATH_CUDA=gifs/cuda/${grid_size}x${grid_size}/heat_cuda_${grid_size}x${grid_size}_${NUM_STEPS}steps_run${run_num}.gif
        convert -delay 10 -loop 0 $(ls frames_cuda/*.png | sort -V) $GIF_PATH_CUDA
        echo "Generated $GIF_PATH_CUDA."
    fi
}

# Main loop
for grid_size in "${GRID_SIZES[@]}"; do
    create_directories $grid_size

    # Initialize CSV file
    results_csv="results/results_${grid_size}x${grid_size}.csv"
    echo "grid_size,exec_type,run_num,step,time" > $results_csv

    seq_total_time=0
    cuda_total_time=0

    for ((i=0; i<RUNS; i++)); do
        modify_initialize $grid_size $NUM_STEPS $STEP_INTERVAL
        
        # Ensure directories are clean before starting
        make -s iclean

        # Run heat_seq and heat_cuda
        run_simulation $grid_size $((i+1)) "seq"
        run_simulation $grid_size $((i+1)) "cuda"

        make -s iclean
    done

    echo "Total seq time for grid size ${grid_size}: $seq_total_time seconds"
    echo "Total cuda time for grid size ${grid_size}: $cuda_total_time seconds"

    # Generate graphs for current grid size
    make -s graph_${grid_size}
done

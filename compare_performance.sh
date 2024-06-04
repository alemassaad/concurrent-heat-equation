#!/bin/bash

# Define main variables
GRID_SIZES=(1000 1200)
NUM_STEPS=($(seq 100 100 300))
RUNS=2
STEP_INTERVAL=100

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

# Create results directory
mkdir -p results

# Iterate over grid sizes and number of steps
for size in "${GRID_SIZES[@]}"; do

    mkdir -p gifs/seq/${size}x${size} gifs/cuda/${size}x${size}
    
    # Initialize CSV file
    results_csv="results/results_${size}x${size}.csv"
    echo "grid_size,num_steps,average_seq_time,average_cuda_time" > $results_csv

    for num in "${NUM_STEPS[@]}"; do

        seq_total_time=0
        cuda_total_time=0
        
        for ((i=0; i<RUNS; i++)); do

            modify_initialize $size $num $STEP_INTERVAL
            
            # Run heat_seq
            start_time=$(date +%s%N)
            make -s heat_seq
            end_time=$(date +%s%N)
            make -s generate_frames_seq
            seq_total_time=$(echo "$seq_total_time + ($end_time - $start_time) / 1000000000" | bc -l)

            # Run gnuplot for seq
            gnuplot plot_seq.gp

            # Create GIF for seq
            GIF_PATH_SEQ=gifs/seq/${size}x${size}/heat_seq_${size}x${size}_${num}steps_run${i}.gif
            convert -delay 10 -loop 0 heatmaps_seq/*.png $GIF_PATH_SEQ
            echo "Generated $GIF_PATH_SEQ."

            # Run heat_cuda
            start_time=$(date +%s%N)
            make -s heat_cuda
            end_time=$(date +%s%N)
            make -s generate_frames_cuda
            cuda_total_time=$(echo "$cuda_total_time + ($end_time - $start_time) / 1000000000" | bc -l)

            # Run gnuplot for cuda
            gnuplot plot_cuda.gp

            # Create GIF for cuda
            GIF_PATH_CUDA=gifs/cuda/${size}x${size}/heat_cuda_${size}x${size}_${num}steps_run${i}.gif
            convert -delay 10 -loop 0 heatmaps_cuda/*.png $GIF_PATH_CUDA
            echo "Generated $GIF_PATH_CUDA."
        
            make -s iclean
        done
        
        # Calculate average times
        seq_avg_time=$(echo "scale=2; $seq_total_time / $RUNS" | bc)
        cuda_avg_time=$(echo "scale=2; $cuda_total_time / $RUNS" | bc)

        # Save to CSV
        echo "$size,$num,$seq_avg_time,$cuda_avg_time" >> $results_csv
    done
done

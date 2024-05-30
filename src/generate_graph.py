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


import matplotlib.pyplot as plt
import os

# Define grid size and number of steps
grid_size = 300
num_steps = [100,300,500,700,900,1100,1300,1500,1700,1900]

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


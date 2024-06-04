import matplotlib.pyplot as plt
import os
import sys
import csv

# Get the grid size from command-line arguments
if len(sys.argv) != 2:
    print("Usage: python3 generate_graphs.py <grid_size>")
    sys.exit(1)

grid_size = sys.argv[1]

# Create a results directory for the graphs
os.makedirs("graphs", exist_ok=True)

# Read the CSV file for the current grid size
csv_file = f"results/results_{grid_size}x{grid_size}.csv"

num_steps = []
average_seq_time = []
average_cuda_time = []

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        num_steps.append(int(row[1]))
        average_seq_time.append(float(row[2]))
        average_cuda_time.append(float(row[3]))

# Create a line plot
plt.figure(figsize=(10, 6))
plt.plot(num_steps, average_seq_time, marker='o', label='Sequential')
plt.plot(num_steps, average_cuda_time, marker='o', label='CUDA')

# Title and labels
plt.title(f'Performance Comparison for Grid Size {grid_size}x{grid_size}')
plt.xlabel('Number of Steps')
plt.ylabel('Time (s)')
plt.legend()

# Save the plot
plt.savefig(f"graphs/performance_comparison_{grid_size}x{grid_size}.png")
plt.close()

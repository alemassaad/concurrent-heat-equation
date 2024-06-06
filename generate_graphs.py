import matplotlib.pyplot as plt
import os
import sys
import csv
from collections import defaultdict

# Get the grid sizes from command-line arguments
if len(sys.argv) < 2:
    print("Usage: python3 generate_graphs.py <grid_size1> <grid_size2> ...")
    sys.exit(1)

grid_sizes = sys.argv[1:]

for grid_size in grid_sizes:
    # Create a results directory for the graphs of the current grid size
    os.makedirs(f"graphs/{grid_size}", exist_ok=True)

    # Read the CSV file for the current grid size
    csv_file = f"results/results_{grid_size}x{grid_size}.csv"

    steps = defaultdict(list)
    seq_times = defaultdict(list)
    cuda_times = defaultdict(list)

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            step = int(row[3])
            time = float(row[4])
            if row[1] == 'seq':
                seq_times[step].append(time)
            elif row[1] == 'cuda':
                cuda_times[step].append(time)
            steps[step].append(time)

    steps_sorted = sorted(steps.keys())
    cum_seq_times = [sum(seq_times[step]) / len(seq_times[step]) for step in steps_sorted]
    cum_cuda_times = [sum(cuda_times[step]) / len(cuda_times[step]) for step in steps_sorted]

    # Calculate relative performance and percentage improvement
    relative_performance = [seq_time / cuda_time for seq_time, cuda_time in zip(cum_seq_times, cum_cuda_times)]
    percentage_improvement = [(1 - (cuda_time / seq_time)) * 100 for seq_time, cuda_time in zip(cum_seq_times, cum_cuda_times)]
    inverse_percentage_improvement = [100 - pi for pi in percentage_improvement]

    # Create a line plot for cumulative time (log scale)
    plt.figure(figsize=(10, 6))
    plt.plot(steps_sorted, cum_seq_times, 'b-', marker='o', label='Sequential')
    plt.plot(steps_sorted, cum_cuda_times, 'r-', marker='o', label='CUDA')

    # Title and labels
    plt.title(f'Performance Comparison for Grid Size {grid_size}x{grid_size} (Log Scale)')
    plt.xlabel('Number of Steps')
    plt.ylabel('Cumulative Time (s)')
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend()

    # Save the cumulative time plot
    plt.savefig(f"graphs/{grid_size}/performance_comparison_{grid_size}x{grid_size}_logscale.png")
    plt.close()

    # Create a plot for relative performance
    plt.figure(figsize=(10, 6))
    plt.plot(steps_sorted, relative_performance, 'g-', marker='o', label='Relative Performance (Seq/CUDA)')
    
    # Title and labels
    plt.title(f'Relative Performance for Grid Size {grid_size}x{grid_size}')
    plt.xlabel('Number of Steps')
    plt.ylabel('Relative Performance (Seq/CUDA)')
    plt.legend()

    # Save the relative performance plot
    plt.savefig(f"graphs/{grid_size}/relative_performance_{grid_size}x{grid_size}.png")
    plt.close()

    # Create a plot for percentage improvement (log scale)
    plt.figure(figsize=(10, 6))
    plt.plot(steps_sorted, inverse_percentage_improvement, 'm-', marker='o', label='Inverse Percentage Improvement (100 - %)')
    
    # Title and labels
    plt.title(f'Inverse Percentage Improvement for Grid Size {grid_size}x{grid_size} (Log Scale)')
    plt.xlabel('Number of Steps')
    plt.ylabel('Inverse Percentage Improvement (100 - %)')
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend()

    # Save the inverse percentage improvement plot
    plt.savefig(f"graphs/{grid_size}/inverse_percentage_improvement_{grid_size}x{grid_size}_logscale.png")
    plt.close()

print("Graphs generated successfully.")

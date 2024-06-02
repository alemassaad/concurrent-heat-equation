# Extract N_STEPS and STEP_INTERVAL from initialize.h
N_STEPS=$(shell grep 'define N_STEPS' initialize.h | awk '{print $$3}')
STEP_INTERVAL=$(shell grep 'define STEP_INTERVAL' initialize.h | awk '{print $$3}')

all: run_seq run_cuda create_gifs

heat_seq: heat_seq.c
	@gcc -o heat_seq heat_seq.c -lm

heat_cuda: heat_cuda.cu
	@nvcc -o heat_cuda heat_cuda.cu

run_seq: heat_seq
	@mkdir -p output_seq heatmaps_seq
	@echo "Running heat_seq..."
	@./heat_seq
	@echo "Generating gnuplot script for seq..."
	@./generate_gnuplot_script.sh seq $(N_STEPS) $(STEP_INTERVAL)
	@echo "Running gnuplot for seq..."
	@gnuplot plot_seq.gp
	@echo "Sequential run completed."

run_cuda: heat_cuda
	@mkdir -p output_cuda heatmaps_cuda
	@echo "Running heat_cuda..."
	@./heat_cuda
	@echo "Generating gnuplot script for cuda..."
	@./generate_gnuplot_script.sh cuda $(N_STEPS) $(STEP_INTERVAL)
	@echo "Running gnuplot for cuda..."
	@gnuplot plot_cuda.gp
	@echo "CUDA run completed."

create_gifs:
	@echo "Creating GIFs..."
	@convert -delay 10 -loop 0 heatmaps_seq/*.png heatmap_seq.gif
	@convert -delay 10 -loop 0 heatmaps_cuda/*.png heatmap_cuda.gif
	@echo "GIFs created."

performance:
	@chmod +x compare_performance.sh
	@./compare_performance.sh

clean:
	@rm -f heat_seq heat_cuda
	@rm -rf output_seq output_cuda heatmaps_seq heatmaps_cuda
	@rm -f heatmap_seq.gif heatmap_cuda.gif
	@rm -rf results
	@rm -f performance_comparison_*.png
	@rm -f generate_graph_*.py
	@rm -f plot_seq.gp plot_cuda.gp
	@rm -rf gifs

.PHONY: clean run_seq run_cuda create_gifs performance

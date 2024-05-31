all: run_seq run_cuda create_gifs compare_outputs

heat_seq: heat_seq.c
	@gcc -o heat_seq heat_seq.c -lm

heat_cuda: heat_cuda.cu
	@nvcc -o heat_cuda heat_cuda.cu

run_seq: heat_seq
	@mkdir -p output_seq heatmaps_seq
	@./heat_seq
	@gnuplot plot_seq.gp

run_cuda: heat_cuda
	@mkdir -p output_cuda heatmaps_cuda
	@./heat_cuda
	@gnuplot plot_cuda.gp

create_gifs:
	@convert -delay 10 -loop 0 heatmaps_seq/*.png heatmap_seq.gif
	@convert -delay 10 -loop 0 heatmaps_cuda/*.png heatmap_cuda.gif

compare_performance:
	@chmod +x compare_performance.sh
	@./compare_performance.sh

compare_outputs:
	@chmod +x compare_outputs.sh
	@./compare_outputs.sh

clean:
	@rm -f heat_seq heat_cuda
	@rm -rf output_seq output_cuda heatmaps_seq heatmaps_cuda
	@rm -f heatmap_seq.gif heatmap_cuda.gif
	@rm -rf results
	@rm -f performance_comparison_*.png
	@rm -f generate_graph_*.py

.PHONY: clean run_seq run_cuda create_gifs compare_performance compare_outputs

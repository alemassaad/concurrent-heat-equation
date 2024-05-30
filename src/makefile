all: run_seq run_parallel create_gifs

heat_seq: heat_seq.c
	@gcc -o heat_seq heat_seq.c -lm

heat_parallel: heat_parallel.cu
	@nvcc -o heat_parallel heat_parallel.cu

run_seq: heat_seq
	@mkdir -p output_seq heatmaps_seq
	@./heat_seq
	@gnuplot plot_seq.gp

run_parallel: heat_parallel
	@mkdir -p output_parallel heatmaps_parallel
	@./heat_parallel
	@gnuplot plot_parallel.gp

create_gifs:
	@convert -delay 10 -loop 0 heatmaps_seq/*.png heatmap_seq.gif
	@convert -delay 10 -loop 0 heatmaps_parallel/*.png heatmap_parallel.gif

clean:
	@rm -f heat_seq heat_parallel output_seq/*.dat output_parallel/*.dat heatmaps_seq/*.png heatmaps_parallel/*.png heatmap_seq.gif heatmap_parallel.gif

.PHONY: clean run_seq run_parallel create_gifs

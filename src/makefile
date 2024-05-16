all: heat_seq run_plot create_gif

heat_seq: heat_seq.c
	@gcc -o heat_seq heat_seq.c -lm

run_plot: heat_seq
	@mkdir -p output heatmaps
	@./heat_seq
	@gnuplot plot.gp

create_gif:
	@convert -delay 10 -loop 0 heatmaps/*.png heatmap_animation.gif

clean:
	@rm -f heat_seq output/*.dat heatmaps/*.png heatmap_animation.gif

.PHONY: clean run_plot create_gif

GRID_SIZES := $(shell grep 'GRID_SIZES' compare_performance.sh | awk -F'[()]' '{print $$2}' | tr ',' ' ')
N_STEPS := $(shell grep 'define N_STEPS' initialize.h | awk '{print $$3}')
STEP_INTERVAL=$(shell grep 'define STEP_INTERVAL' initialize.h | awk '{print $$3}')
NX=$(shell grep 'define NX' initialize.h | awk '{print $$3}')
NY=$(shell grep 'define NY' initialize.h | awk '{print $$3}')

all: run_seq run_cuda generate_gif_seq generate_gif_cuda

compile_seq: heat_seq.c
	@echo "Compiling heat_seq.c"
	@gcc -o heat_seq heat_seq.c -lm
	@echo "Compilation done: created heat_seq"
	
compile_cuda: heat_cuda.cu
	@echo "Compiling heat_cuda.cu"
	@nvcc -o heat_cuda heat_cuda.cu
	@echo "Compilation done: created heat_cuda"

execute_seq:
	@mkdir -p output_seq frames_seq
	@echo "Executing heat_seq with run number $(run_num)"
	@./heat_seq $(run_num)
	@echo "SEQ Execution done"


execute_cuda:
	@mkdir -p output_cuda frames_cuda
	@echo "Executing heat_cuda with run number $(run_num)"
	@./heat_cuda $(run_num)
	@echo "CUDA Execution done"

generate_frames_seq:
	@./generate_gnuplot.sh seq $(N_STEPS) $(STEP_INTERVAL) $(NX) $(NY)
	@gnuplot plot_seq.gp

generate_frames_cuda:
	@./generate_gnuplot.sh cuda $(N_STEPS) $(STEP_INTERVAL) $(NX) $(NY)
	@gnuplot plot_cuda.gp

run_seq: compile_seq execute_seq generate_frames_seq
	@echo "Sequential run completed"

run_cuda: compile_cuda execute_cuda generate_frames_cuda
	@echo "CUDA run completed"

generate_gif_seq:
	@echo "Generating SEQ GIF"
	@convert -delay 10 -loop 0 $(shell ls frames_seq/*.png | sort -V) heatmap_seq.gif
	@echo "SEQ GIF generated"

generate_gif_cuda:
	@echo "Generating CUDA GIF"
	@convert -delay 10 -loop 0 $(shell ls frames_cuda/*.png | sort -V) heatmap_cuda.gif
	@echo "CUDA GIF generated"

performance:
	@chmod +x compare_performance.sh
	@./compare_performance.sh

generate_graphs:
	@$(MAKE) -s graph_all

graph_all: $(patsubst %, graph_%, $(GRID_SIZES))

graph_%:
	@echo "Generating graphs for grid size $*"
	@mkdir -p graphs/$*
	@python3 generate_graphs.py $*

clean:
	@rm -f heat_seq heat_cuda plot_seq.gp plot_cuda.gp heatmap_seq.gif heatmap_cuda.gif
	@rm -rf output_seq/* output_cuda/* frames_seq/* frames_cuda/* results/* gifs/* graphs/*

iclean: 
	@rm -f heat_seq heat_cuda plot_seq.gp plot_cuda.gp
	@rm -rf frames_seq/* frames_cuda/* output_seq/* output_cuda/*

.PHONY: clean run_seq run_cuda generate_gif_seq generate_gif_cuda performance

#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 [seq|cuda] [N_STEPS] [STEP_INTERVAL] [NX] [NY]"
    exit 1
fi

MODE=$1
N_STEPS=$2
STEP_INTERVAL=$3
NX=$4
NY=$5

if [ "$MODE" == "seq" ]; then
    OUTPUT_DIR="output_seq"
    GNUPLOT_SCRIPT="plot_seq.gp"
    FRAMES_DIR="frames_seq"
elif [ "$MODE" == "cuda" ]; then
    OUTPUT_DIR="output_cuda"
    GNUPLOT_SCRIPT="plot_cuda.gp"
    FRAMES_DIR="frames_cuda"
else
    echo "Invalid mode: $MODE"
    exit 1
fi

cat <<EOF > $GNUPLOT_SCRIPT
set terminal png size 800,800
set pm3d map
set palette defined (0 "blue", 1 "green", 2 "yellow", 3 "red")
set cbrange [-100:100]  # Adjust color range to include negative values
set xrange [0:$(($NX-1))]
set yrange [0:$(($NY-1))]
unset key
set view map

EOF

for ((i=0; i<=N_STEPS; i+=STEP_INTERVAL)); do
    FILENAME="$OUTPUT_DIR/output_$i.dat"
    if [ -f "$FILENAME" ]; then
        echo "set output sprintf('$FRAMES_DIR/frame_%04d.png', $i)" >> $GNUPLOT_SCRIPT
        echo "splot sprintf('$OUTPUT_DIR/output_%d.dat', $i) matrix with image" >> $GNUPLOT_SCRIPT
    fi
done

# Execute the gnuplot script
gnuplot $GNUPLOT_SCRIPT

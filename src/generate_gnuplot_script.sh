#!/bin/bash

if [ "$1" == "seq" ]; then
    OUTPUT_DIR="output_seq"
    GNUPLOT_SCRIPT="plot_seq.gp"
    HEATMAPS_DIR="heatmaps_seq"
elif [ "$1" == "cuda" ]; then
    OUTPUT_DIR="output_cuda"
    GNUPLOT_SCRIPT="plot_cuda.gp"
    HEATMAPS_DIR="heatmaps_cuda"
else
    echo "Usage: $0 [seq|cuda] [N_STEPS] [STEP_INTERVAL]"
    exit 1
fi

N_STEPS=${2:-60000}
STEP_INTERVAL=${3:-100}

echo "Generating $GNUPLOT_SCRIPT for $1 with N_STEPS=$N_STEPS and STEP_INTERVAL=$STEP_INTERVAL"

cat <<EOF > $GNUPLOT_SCRIPT
set terminal png size 800,800
set pm3d map
set palette defined (0 "blue", 1 "green", 2 "yellow", 3 "red")
set cbrange [0:100]
set xrange [0:99]
set yrange [0:99]
unset key
set view map

EOF

for ((i=0; i<=N_STEPS; i+=STEP_INTERVAL)); do
    FILENAME="$OUTPUT_DIR/output_$i.dat"
    if [ -f "$FILENAME" ]; then
        echo "set output sprintf('$HEATMAPS_DIR/heatmap_%04d.png', $i)" >> $GNUPLOT_SCRIPT
        echo "splot sprintf('$OUTPUT_DIR/output_%d.dat', $i) matrix with image" >> $GNUPLOT_SCRIPT
    fi
done

echo "$GNUPLOT_SCRIPT generated successfully."

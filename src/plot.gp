set terminal png size 800,800
set pm3d map
set palette defined (0 "blue", 1 "green", 2 "yellow", 3 "red")
set cbrange [0:100]  # Adjust based on expected range
set xrange [0:99]
set yrange [0:99]
unset key
set view map

do for [i=0:599:1] {
    set output sprintf('heatmaps/heatmap_%04d.png', i)
    splot sprintf('output/output_%d.dat', i) matrix with image
}

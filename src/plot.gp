set terminal png size 800,800
set output 'heatmap.png'
set pm3d map
set palette defined (0 "blue", 1 "white", 2 "red")
set cbrange [0:100]  # Adjust depending on the range of your data
set xrange [0:99]    # Adjust if different number of columns
set yrange [0:99]    # Adjust if different number of rows
splot 'output.dat' matrix nonuniform
pause -1

#!/bin/bash

# Directory paths
seq_dir="output_seq"
cuda_dir="output_cuda"

# Number of steps
n_steps=600
tolerance=0.1  # Set an acceptable tolerance level

# Check for debug mode
DEBUG=false
if [ "$1" == "debug" ]; then
    DEBUG=true
fi

within_tolerance=true

# Loop through all steps and compare files
for (( i=0; i<n_steps; i++ ))
do
    seq_file="$seq_dir/output_$i.dat"
    cuda_file="$cuda_dir/output_$i.dat"
    
    if diff "$seq_file" "$cuda_file" > /dev/null; then
        if [ "$DEBUG" = true ]; then
            echo "Step $i: Files are identical"
        fi
    else
        if [ "$DEBUG" = true ]; then
            echo "Step $i: Files differ"
        fi
        
        # Compare values within the tolerance level
        awk -v tol="$tolerance" -v step="$i" -v debug="$DEBUG" '
        NR == FNR { a[NR] = $0; next }
        {
            split(a[FNR], b, " ")
            split($0, c, " ")
            for (i = 1; i <= length(b); i++) {
                diff = b[i] - c[i]
                if (diff < 0) diff = -diff
                if (diff > tol) {
                    if (debug == "true") {
                        printf "Step %d: Difference at line %d, value %f (seq) vs %f (cuda)\n", step, FNR, b[i], c[i]
                    }
                    exit 1
                }
            }
        }' "$seq_file" "$cuda_file"
        
        if [ $? -ne 0 ]; then
            within_tolerance=false
        fi
    fi
done

if [ "$within_tolerance" = true ]; then
    echo "All steps are within the tolerance range of $tolerance."
else
    echo "Some steps have differences beyond the tolerance range of $tolerance."
fi

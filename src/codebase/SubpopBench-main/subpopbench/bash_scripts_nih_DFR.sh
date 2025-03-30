#!/bin/bash

# Define the directory containing the .sh files
DIR="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/subpopbench/nih_scripts_DFR"

# Change to the directory
cd "$DIR"

# Loop over each .sh file in the directory and submit it
for file in *.sh; do
    echo "Submitting $file..."
    sbatch -A bio170034p -p BatComputer --gres=gpu:rtx6000:1 -N 1 -t 48:00:00 "$file"
done

echo "All jobs submitted."

#!/bin/bash
#SBATCH --account=cnncae
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=didymus
#SBATCH --exclusive
#SBATCH -o job.bm.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

for i in {26..40}; do
    echo "Processing STL number: $i" >> job.bm.out
    /home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- $i &
done

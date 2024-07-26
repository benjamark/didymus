#!/bin/bash
#SBATCH --account=cnncae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
# #SBATCH --mem-per-cpu=2G
#SBATCH --job-name=didymus
#SBATCH --exclusive
#SBATCH -o job2.cpu.out


source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

#python3 reconstruct_stl.py
#python3 npys_to_stls.py

for i in $(seq 0 3); do
    echo "Processing STL number: $i" >> job2.cpu.out
    /home/markben/software/blender-3.3.0-linux-x64/blender -b -P remesh2.py -- $i &
done

wait

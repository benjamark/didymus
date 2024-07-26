#!/bin/bash
#SBATCH --account=cnncae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#  #SBATCH --mem-per-cpu=16G
#SBATCH --job-name=didymus
#SBATCH -o job.cpu.out
#SBATCH --partition=debug


source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

python3 extrapolate_stl.py

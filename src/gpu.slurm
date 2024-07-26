#!/bin/bash
#SBATCH --account=cnncae
#SBATCH --reservation=h100-testing
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --job-name=didymus
#SBATCH --exclusive
#SBATCH -o job.gpu.out

export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc
ml conda
conda activate numba

srun python3 build_boolean_surface.py

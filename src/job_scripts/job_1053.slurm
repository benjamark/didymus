#!/bin/bash
#SBATCH --account=cnncae
#SBATCH --reservation=h100-testing
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=didymus_1053
#SBATCH --exclusive
# #SBATCH --mem-per-cpu=64G
#SBATCH -o job_scripts/job_1053.cpu.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

echo "Processing STL number: 1053" >> job_1053.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 1053 &
echo "Processing STL number: 1054" >> job_1053.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 1054 &
echo "Processing STL number: 1055" >> job_1053.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 1055 &
wait

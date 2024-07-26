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
#SBATCH --job-name=didymus_75
#SBATCH --exclusive
# #SBATCH --mem-per-cpu=64G
#SBATCH -o job_scripts/job_75.cpu.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

echo "Processing STL number: 75" >> job_75.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 75 &
echo "Processing STL number: 76" >> job_75.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 76 &
echo "Processing STL number: 77" >> job_75.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 77 &
wait

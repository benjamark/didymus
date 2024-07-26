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
#SBATCH --job-name=didymus_468
#SBATCH --exclusive
# #SBATCH --mem-per-cpu=64G
#SBATCH -o job_scripts/job_468.cpu.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

echo "Processing STL number: 468" >> job_468.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 468 &
echo "Processing STL number: 469" >> job_468.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 469 &
echo "Processing STL number: 470" >> job_468.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 470 &
wait

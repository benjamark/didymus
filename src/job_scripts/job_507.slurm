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
#SBATCH --job-name=didymus_507
#SBATCH --exclusive
# #SBATCH --mem-per-cpu=64G
#SBATCH -o job_scripts/job_507.cpu.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

echo "Processing STL number: 507" >> job_507.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 507 &
echo "Processing STL number: 508" >> job_507.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 508 &
echo "Processing STL number: 509" >> job_507.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 509 &
wait

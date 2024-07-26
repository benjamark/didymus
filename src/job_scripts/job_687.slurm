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
#SBATCH --job-name=didymus_687
#SBATCH --exclusive
# #SBATCH --mem-per-cpu=64G
#SBATCH -o job_scripts/job_687.cpu.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

echo "Processing STL number: 687" >> job_687.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 687 &
echo "Processing STL number: 688" >> job_687.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 688 &
echo "Processing STL number: 689" >> job_687.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 689 &
wait

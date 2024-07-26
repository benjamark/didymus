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
#SBATCH --job-name=didymus_966
#SBATCH --exclusive
# #SBATCH --mem-per-cpu=64G
#SBATCH -o job_scripts/job_966.cpu.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

echo "Processing STL number: 966" >> job_966.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 966 &
echo "Processing STL number: 967" >> job_966.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 967 &
echo "Processing STL number: 968" >> job_966.cpu.out
/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- 968 &
wait

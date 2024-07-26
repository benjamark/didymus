import os

slurm_template = """#!/bin/bash
#SBATCH --account=cnncae
#SBATCH --reservation=h100-testing
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --job-name=didymus_{idx}
#SBATCH --exclusive
#SBATCH -o nts_job_{idx}.gpu.out

export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc
ml conda
conda activate numba

python3 one_npy_to_stl_ext.py {idx}
"""

output_dir = "."

for i in range(300):
    slurm_script = slurm_template.format(idx=i)
    with open(os.path.join(output_dir, f"nts_job_{i}.sh"), "w") as file:
        file.write(slurm_script)

# create a script to sbatch all job scripts
with open(os.path.join(output_dir, "submit_all_n_t_s_ext_jobs.sh"), "w") as submit_file:
    submit_file.write("#!/bin/bash\n\n")
    for i in range(300):
        submit_file.write(f"sbatch nts_job_{i}.sh\n")

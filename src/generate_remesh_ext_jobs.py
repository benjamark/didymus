import os

slurm_template = """#!/bin/bash
#SBATCH --account=cnncae
#SBATCH --reservation=h100-testing
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=didymus_{idx}
#SBATCH --exclusive
#SBATCH -o remesh_{idx}.gpu.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo_ext.py -- {idx}
"""

output_dir = "."
os.makedirs(output_dir, exist_ok=True)

for i in range(300):
    slurm_script = slurm_template.format(idx=i)
    with open(os.path.join(output_dir, f"remesh_job_{i}.sh"), "w") as file:
        file.write(slurm_script)

# create a script to sbatch all job scripts
with open(os.path.join(output_dir, "submit_all_remesh_ext_jobs.sh"), "w") as submit_file:
    submit_file.write("#!/bin/bash\n\n")
    for i in range(300):
        submit_file.write(f"sbatch remesh_job_{i}.sh\n")


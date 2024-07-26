#!/bin/bash
# This script will generate and submit SLURM jobs, each handling 3 cases

TOTAL_STL_FILES=1128
BATCH_SIZE=3
JOB_SCRIPT_DIR="./job_scripts"  # Directory to store the job scripts

mkdir -p $JOB_SCRIPT_DIR

for ((i = 0; i < TOTAL_STL_FILES; i += 4)); do
    end=$((i + 4 * (BATCH_SIZE - 1)))
    if [ $end -ge $TOTAL_STL_FILES ]; then
        end=$((TOTAL_STL_FILES - 1))
    fi

    job_script="$JOB_SCRIPT_DIR/job_$i.slurm"

    cat <<EOT > $job_script
#!/bin/bash
#SBATCH --account=cnncae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --job-name=didymus_$i
#SBATCH --exclusive
#SBATCH -o small_job_$i.cpu.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

EOT

    for j in $(seq $i 4 $end); do
        echo "echo \"Processing STL number: $j\" >> small_job_$i.cpu.out" >> $job_script
        echo "/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- $j &" >> $job_script
    done

    echo "wait" >> $job_script

    sbatch $job_script
done


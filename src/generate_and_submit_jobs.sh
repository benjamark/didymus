#!/bin/bash
# generate and submit SLURM jobs, each handling 3 cases

TOTAL_STL_FILES=1275
BATCH_SIZE=3
JOB_SCRIPT_DIR="./job_scripts"  # Directory to store the job scripts

mkdir -p $JOB_SCRIPT_DIR

for ((i = 0; i < TOTAL_STL_FILES; i += BATCH_SIZE)); do
    end=$((i + BATCH_SIZE - 1))
    if [ $end -ge $TOTAL_STL_FILES ]; then
        end=$((TOTAL_STL_FILES - 1))
    fi

    job_script="$JOB_SCRIPT_DIR/job_$i.slurm"

    cat <<EOT > $job_script
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
#SBATCH --job-name=didymus_$i
#SBATCH --exclusive
# #SBATCH --mem-per-cpu=64G
#SBATCH -o job_scripts/job_$i.cpu.out

source ~/.bashrc
ml conda
conda activate numba
ml gcc/13.1.0  # needed for npys_to_stls.py

EOT

    for j in $(seq $i $end); do
        echo "echo \"Processing STL number: $j\" >> job_$i.cpu.out" >> $job_script
        echo "/home/markben/software/blender-3.3.0-linux-x64/blender -b -P combo.py -- $j &" >> $job_script
    done

    echo "wait" >> $job_script

    sbatch $job_script
done

#!/bin/bash

main_dir=##################
device=a100

log_dir="${main_dir}/logs"
runs_dir="${main_dir}/runs"

mkdir -p "${log_dir}"
mkdir -p "${runs_dir}"
CUDA_VISIBLE_DEVICES="0"
module add CUDA
echo "Starting batch job submission..."
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running as user: $(whoami)"
echo "Start time: $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

batch_sizes=(10)
alphas=(0.9)
for batch_size in "${batch_sizes[@]}"; do
    for alpha in "${alphas[@]}"; do
    echo -e "\nSubmitting job for epochs: batch_size: ${batch_size}"
    echo -e "Running commands on: $(hostname)"
    echo -e "Start time: $(date '+%F %H:%M:%S')"
    echo $CUDA_VISIBLE_DEVICES

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=llm_calib_${batch_size}
#SBATCH --output=${log_dir}/${alpha}_%j.stdout
#SBATCH --error=${log_dir}/${alpha}_%j.stderr
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=160000
#SBATCH --time=04:40:01
#SBATCH --cpus-per-task=8
#SBATCH --partition=############
#SBATCH --account=##################
#SBATCH --mail-user=##############
#SBATCH --mail-type=ALL
export MAIN_DIR="${main_dir}"
export BATCH_SIZE="${batch_size}"
export ALPHA="${alpha}"
${main_dir}/evaluate_job.sh
EOF

            echo "Submitted job for epochs: alpha: ${alpha}"
    done
done
echo "Batch job submission completed."

#!/bin/bash

main_dir=#################
device=a100

log_dir="${main_dir}/logs"
runs_dir="${main_dir}/runs"

CUDA_VISIBLE_DEVICES="0,1"
mkdir -p "${log_dir}"
mkdir -p "${runs_dir}"
module add cuda
echo "Starting batch job submission..."
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running as user: $(whoami)"
echo "Start time: $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

epochs_values=(5)
lr_values=(5e-6) #3e-6 5e-6 1e-5 2e-5
batch_sizes=(2)
alphas=(0.0 1.0)

for alpha in "${alphas[@]}"; do
    for epochs in "${epochs_values[@]}"; do
        for lr in "${lr_values[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                echo -e "\nSubmitting job for epochs: ${epochs}, lr: ${lr}, batch_size: ${batch_size}"
                echo -e "Running commands on: $(hostname)"
                echo -e "Start time: $(date '+%F %H:%M:%S')"
                echo $CUDA_VISIBLE_DEVICES

                sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=llm_calib_${epochs}_${lr}_${batch_size}_${alpha}
#SBATCH --output=${log_dir}/alpha${alpha}_epochs${epochs}_lr${lr}_bs${batch_size}_${alpha}%j.stdout
#SBATCH --error=${log_dir}/alpha${alpha}_epochs${epochs}_lr${lr}_bs${batch_size}_${alpha}%j.stderr
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=160G
#SBATCH --time=13:30:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=#######################
#SBATCH --account=#######################
#SBATCH --mail-user=#######################
#SBATCH --mail-type=ALL
export PORT=$((1024+RANDOM % 64512))
export MAIN_DIR="${main_dir}"
export EPOCHS="${epochs}"
export LR="${lr}"
export BATCH_SIZE="${batch_size}"
export ALPHA="${alpha}"
${main_dir}/train_job.sh
EOF
            echo "Submitted job for epochs: ${epochs}, lr: ${lr}, batch_size: ${batch_size}"
            done
        done
    done
done

echo "Batch job submission completed."
echo "Allocated GPU IDs: $CUDA_VISIBLE_DEVICES"
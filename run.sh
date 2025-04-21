#!/bin/bash
#SBATCH --job-name=cs599_rnn_experiments_runner
#SBATCH --output=slurm_output/cs599_rnn_exp_runner_%j.out
#SBATCH --error=slurm_output/cs599_rnn_exp_runner_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB

mkdir -p slurm_output
mkdir -p experiments

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="rnn_comparison_${TIMESTAMP}"

module purge
module load cuda
module load glib

echo "Starting RNN cell experiments at $(date)"
echo "Experiment name: ${EXPERIMENT_NAME}"

echo "Starting RNN cell experiments at $(date)"
echo "Experiment name: ${EXPERIMENT_NAME}"

.venv/bin/python main.py \
    --experiment-name ${EXPERIMENT_NAME} \
    --scheduler slurm

if [ $? -ne 0 ]; then
    echo "Error: The main experiment script failed"
    exit 1
fi

echo "Experiments submitted successfully at $(date)"
echo "Results will be available in: experiments/${EXPERIMENT_NAME}"
echo "You can monitor the progress using: squeue -u $USER"

echo "Script completed at $(date)"
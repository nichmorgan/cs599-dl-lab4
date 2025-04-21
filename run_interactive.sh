#!/bin/bash
# run_interactive.sh - For interactive SLURM runs with logging

# Create directories
mkdir -p slurm_output
mkdir -p experiments

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="rnn_comparison_${TIMESTAMP}"
LOG_FILE="slurm_output/cs599_rnn_exp_runner_${TIMESTAMP}.out"
ERR_FILE="slurm_output/cs599_rnn_exp_runner_${TIMESTAMP}.err"

echo "Starting RNN experiments..."
echo "Log file: ${LOG_FILE}"
echo "Error file: ${ERR_FILE}"

# Run with srun and capture output
srun -n 1 -c 8 --mem=16GB --pty bash -c "
    set -e
    
    # Load modules and activate environment
    module purge
    module load anaconda3
    conda activate .venv/
    
    # Set environment variables
    export MPLBACKEND=Agg
    export TF_CPP_MIN_LOG_LEVEL=2
    
    echo 'Starting RNN cell experiments at $(date)'
    echo 'Experiment name: ${EXPERIMENT_NAME}'
    
    # Run the main script with tee for dual output
    python main.py \
        --experiment-name ${EXPERIMENT_NAME} \
        --scheduler slurm 2>&1 | tee -a '${LOG_FILE}'
    
    echo 'Script completed at $(date)'
" 2>&1 | tee -a "${ERR_FILE}"

echo "Experiment completed. Check logs in ${LOG_FILE} and ${ERR_FILE}"
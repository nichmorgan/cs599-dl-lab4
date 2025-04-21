#!/bin/bash
#SBATCH --job-name=cs599_rnn_experiments_runner
#SBATCH --output=slurm_output/cs599_rnn_exp_runner_%j.out
#SBATCH --error=slurm_output/cs599_rnn_exp_runner_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB

set -e  # Exit immediately if a command exits with a non-zero status

mkdir -p slurm_output
mkdir -p experiments
mkdir -p $HOME/.local/lib/python3.11/site-packages  # Ensure local install directory exists

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="rnn_comparison_${TIMESTAMP}"

# Ensure clean environment
module purge
module load anaconda3

conda activate .venv/

# Set environment variables
export MPLBACKEND=Agg  # Force matplotlib to use the 'Agg' backend without X11
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging noise

echo "Starting RNN cell experiments at $(date)"
echo "Experiment name: ${EXPERIMENT_NAME}"

# Verify installation
echo "Verifying installation..."
python -c "import matplotlib; matplotlib.use('Agg'); from matplotlib import pyplot as plt; print('Matplotlib-base successfully installed')"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install or import matplotlib-base"
    exit 1
fi

# Run the experiment
echo "Starting experiment..."
python main.py \
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
# CS 599: Foundations of Deep Learning - Assignment 4

This repository contains implementations of various Recurrent Neural Network (RNN) cells using TensorFlow 2.x operations, as required for Assignment 4. The implementation uses Dask for parallel execution, allowing experiments to run on either local machines or SLURM-based HPC clusters like Monsoon.

## Project Structure

- `main.py`: Main entry point with Dask integration and command-line argument parsing
- `data_loader.py`: Functions for loading and preprocessing the notMNIST dataset
- `models.py`: Model creation functions
- `training.py`: Training and evaluation functions with tqdm progress bars
- `visualization.py`: Functions for plotting results
- `experiment.py`: Functions for running individual experiment configurations
- `gru_cell.py`: Implementation of Gated Recurrent Unit (GRU)
- `mgu_cell.py`: Implementation of Minimal Gated Unit (MGU)
- `lstm_cell.py`: Implementation of LSTM cell

## Setup Instructions

### Environment Setup

Create and activate the conda environment using the provided `requirements.yaml` file:

```bash
# Create the environment
conda env create -f requirements.yaml

# Activate the environment
conda activate cs599-dl-lab4
```

### Running Experiments

#### Local Execution

To run experiments on your local machine:

```bash
python main.py --scheduler local --workers 4
```

#### HPC Execution with SLURM

To run experiments on a SLURM-based HPC cluster like Monsoon:

```bash
python main.py --scheduler slurm
```

#### Customizing Configurations

You can customize the experiment configurations:

```bash
python main.py --model-types gru mgu --hidden-units 128 256 --trials 3 --epochs 10
```

## Command-Line Arguments

The `main.py` script accepts the following arguments:

### Experiment Configuration

- `--experiment-name`: Name for the experiment (default: timestamp)
- `--output-dir`: Directory to save experiment results (default: experiments)

### Model Configuration

- `--model-types`: Model types to evaluate (choices: gru, mgu, lstm)
- `--hidden-units`: List of hidden unit sizes to test

### Training Configuration

- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--trials`: Number of trials for each configuration (default: 3)
- `--log-interval`: Number of batches between logging updates (default: 10)

### Dask Configuration

- `--scheduler`: Dask scheduler to use (choices: local, slurm)
- `--workers`: Number of workers for local scheduler (default: 4)
- `--slurm-cpus`: CPUs per SLURM task (default: 8)
- `--slurm-mem`: Memory per SLURM task (default: 16GB)
- `--slurm-time`: Time limit per SLURM task (default: 08:00:00)

## Debugging in VSCode

The repository includes a `.vscode/launch.json` file with multiple configurations for debugging:

- **Python: Main (Local)**: Debug a local run with minimal settings
- **Python: Main (SLURM)**: Debug a SLURM configuration
- **Python: Single Config (Testing)**: Run a minimal configuration for quick testing
- **Python: Data Loader Test**: Test just the data loading functionality
- **Python: Current File**: Debug the currently open file

## Output Structure

The experiments generate the following output structure:

```
experiments/
└── experiment_name/
    ├── config.json             # Experiment configuration
    ├── summary_report.txt      # Summary of all results
    ├── model_comparison_*.png  # Comparison plots
    └── model_type_units/       # Each configuration
        ├── average_*.png       # Average plots
        ├── average_history.json # Average history data
        └── trial_N/            # Individual trials
            ├── training_log.txt # Training log
            ├── metrics.jsonl   # Metrics for each epoch
            ├── training_curves.png # Training plots
            └── model_weights/  # Saved model weights
```

## Model Implementations

### Gated Recurrent Unit (GRU)

The GRU cell is implemented according to the update equations:

```
z_t = σ(W_z [s_{t-1}, x_t] + b_z)
r_t = σ(W_r [s_{t-1}, x_t] + b_r)
s~_t = Tanh(W_s[r_t ⊙ s_{t-1}, x_t] + b_s)
s_t = (1 - z_t) ⊙ s_{t-1} + z_t ⊙ s~_t
```

### Minimal Gated Unit (MGU)

The Minimal Gated Unit is implemented according to:

```
f_t = σ(W_f [s_{t-1}, x_t] + b_f)
s~_t = Tanh(W_s[f_t ⊙ s_{t-1}, x_t] + b_s)
s_t = (1 - f_t) ⊙ s_{t-1} + f_t ⊙ s~_t
```

## References

1. Cho, K., Van Merriënboer, B., Bahdanau, D., and Bengio, Y. On the properties of neural machine translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259 (2014).

2. Zhou, G., Wu, J., Zhang, C., and Zhou, Z. Minimal gated unit for recurrent neural networks. CoRR abs/1603.09420 (2016).es. arXiv preprint arXiv:1409.1259 (2014).

2. Zhou, G., Wu, J., Zhang, C., and Zhou, Z. Minimal gated unit for recurrent neural networks. CoRR abs/1603.09420 (2016).
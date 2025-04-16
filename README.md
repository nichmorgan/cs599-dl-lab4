# CS 599: Foundations of Deep Learning - Assignment #4

This repository contains the implementation of various Recurrent Neural Network (RNN) cells using basic TensorFlow 2.x operations, as required for Assignment #4.

## Files Description

- `lstm.py`: Implementation of the LSTM cell using TensorFlow 2.x
- `lstm_cell.py`: Another implementation of LSTM cell with more explicit weight declarations
- `gru_cell.py`: Implementation of the Gated Recurrent Unit (GRU) cell
- `mgu_cell.py`: Implementation of the Minimal Gated Unit (MGU) cell
- `train_and_evaluate.py`: Script to train and evaluate the models on the MNIST dataset
- `requirements.yaml`: Conda environment configuration file

## Environment Setup

### Using Conda (Recommended)

Create and activate the conda environment using the provided `requirements.yaml` file:

```bash
# Create the environment
conda env create -f requirements.yaml

# Activate the environment
conda activate cs599-dl-lab4
```

### Using Pip (Alternative)

Alternatively, you can install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Dataset

The code uses the MNIST dataset, which is automatically loaded using TensorFlow's Keras API.

## Running the Code

To train and compare the GRU and MGU models, run:

```bash
python train_and_evaluate.py
```

By default, this will:
1. Train GRU and MGU models with hidden sizes of 128 and 256 units
2. Run 3 trials for each configuration
3. Train for 10 epochs each
4. Generate training curves and report comparison results

## Modifying Parameters

You can modify the experiment parameters in the `train_and_evaluate.py` file:

```python
# Configuration for experiments
model_types = ['gru', 'mgu']
hidden_units_list = [128, 256]  # You can try different sizes: 50, 128, 256, 512
num_hidden_layers = 1  # Start with 1 layer, can increase up to 4
epochs = 10
trials = 3
```

## Implementation Details

### GRU Implementation

The GRU cell is implemented according to the update equations:
```
z_t = σ(W_z [s_{t-1}, x_t] + b_z)
r_t = σ(W_r [s_{t-1}, x_t] + b_r)
s~_t = Tanh(W_s[r_t ⊙ s_{t-1}, x_t] + b_s)
s_t = (1 - z_t) ⊙ s_{t-1} + z_t ⊙ s~_t
```

### MGU Implementation

The Minimal Gated Unit is implemented according to:
```
f_t = σ(W_f [s_{t-1}, x_t] + b_f)
s~_t = Tanh(W_s[f_t ⊙ s_{t-1}, x_t] + b_s)
s_t = (1 - f_t) ⊙ s_{t-1} + f_t ⊙ s~_t
```

## Results

The training script will automatically generate plots comparing:
1. Training and test accuracy over epochs
2. Training and test loss over epochs

For each model configuration, and a final comparison of all models.

The script will also report:
1. Average training time for each model configuration
2. Final test accuracy for each model
3. Classification error over the 3 trials

## References

1. Cho, K., Van Merriënboer, B., Bahdanau, D., and Bengio, Y. On the properties of neural machine translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259 (2014).
2. Zhou, G., Wu, J., Zhang, C., and Zhou, Z. Minimal gated unit for recurrent neural networks. CoRR abs/1603.09420 (2016).
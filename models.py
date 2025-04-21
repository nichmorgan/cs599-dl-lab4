"""
Model creation and configuration functions.
"""

import tensorflow as tf
import numpy as np

# Import custom RNN cells
try:
    from gru_cell import GRUCell
    from mgu_cell import MGUCell
    from lstm_cell import BasicLSTM_cell
except ImportError:
    print("Warning: RNN cell implementations not found. Make sure the cell files are in the correct path.")


def create_model(model_type, input_shape, hidden_units, num_classes):
    """
    Create RNN model with specified cell type
    
    Args:
        model_type: 'gru', 'mgu', or 'lstm'
        input_shape: Shape of input data (seq_length, feature_dim)
        hidden_units: Number of units in the hidden layer
        num_classes: Number of output classes
        
    Returns:
        model: Created model
    """
    print(f"Creating {model_type.upper()} model with {hidden_units} hidden units...")
    
    if model_type.lower() == 'gru':
        model = GRUCell(input_shape[1], hidden_units, num_classes)
    elif model_type.lower() == 'mgu':
        model = MGUCell(input_shape[1], hidden_units, num_classes)
    elif model_type.lower() == 'lstm':
        model = BasicLSTM_cell(input_shape[1], hidden_units, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Log model parameter count (approximate)
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"Model created with approximately {trainable_params:,} trainable parameters")
    
    return model
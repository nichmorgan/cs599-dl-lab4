"""
Training script for RNN cells on notMNIST dataset
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time

# Import our custom RNN cells
from gru_cell import GRUCell
from mgu_cell import MGUCell

# Set random seeds for reproducibility
tf.random.set_seed(0)
np.random.seed(0)

def load_mnist_data():
    """
    Load MNIST dataset from Keras
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    from tensorflow.keras.datasets import mnist
    
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # Add channel dimension for CNN compatibility
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    print(f"MNIST dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Each image shape: {X_train.shape[1:3]}")
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test, num_classes=10):
    """
    Preprocess the data for RNN training
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_test: Test images
        y_test: Test labels
        num_classes: Number of classes
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    # Reshape images to sequences
    # For RNN, we can treat each row of the image as a timestep
    # (batch_size, height, width, channels) -> (batch_size, height, width*channels)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return X_train, y_train, X_test, y_test

def create_model(model_type, input_shape, hidden_units, num_classes):
    """
    Create RNN model with specified cell type
    
    Args:
        model_type: 'gru' or 'mgu'
        input_shape: Shape of input data (seq_length, feature_dim)
        hidden_units: Number of units in the hidden layer
        num_classes: Number of output classes
        
    Returns:
        model: Created model
    """
    if model_type.lower() == 'gru':
        return GRUCell(input_shape[1], hidden_units, num_classes)
    elif model_type.lower() == 'mgu':
        return MGUCell(input_shape[1], hidden_units, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(model, X_train, y_train, X_test, y_test, 
                batch_size=64, epochs=10, learning_rate=0.001):
    """
    Train the RNN model
    
    Args:
        model: RNN model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        history: Training history
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    
    for epoch in range(epochs):
        # Training loop
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                # Forward pass
                logits, _ = model(x_batch)
                # Use the last timestep's outputs as predictions
                predictions = logits[:, -1, :]
                # Compute loss
                loss_value = loss_fn(y_batch, predictions)
            
            # Compute gradients
            grads = tape.gradient(loss_value, model.trainable_variables)
            # Apply gradients
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Update metrics
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y_batch, predictions)
        
        # End of epoch - collect training metrics
        train_loss = epoch_loss_avg.result()
        train_accuracy = epoch_accuracy.result()
        train_loss_results.append(train_loss)
        train_accuracy_results.append(train_accuracy)
        
        # Evaluation loop
        test_loss_avg = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        for x_test_batch, y_test_batch in test_dataset:
            # Forward pass
            test_logits, _ = model(x_test_batch)
            # Use the last timestep's outputs as predictions
            test_predictions = test_logits[:, -1, :]
            # Compute loss
            test_loss_value = loss_fn(y_test_batch, test_predictions)
            
            # Update metrics
            test_loss_avg.update_state(test_loss_value)
            test_accuracy.update_state(y_test_batch, test_predictions)
        
        # End of epoch - collect test metrics
        test_loss = test_loss_avg.result()
        test_accuracy = test_accuracy.result()
        test_loss_results.append(test_loss)
        test_accuracy_results.append(test_accuracy)
        
        # Print epoch results
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {train_loss:.4f}, "
            f"Accuracy: {train_accuracy:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Accuracy: {test_accuracy:.4f}"
        )
    
    # Return history for plotting
    history = {
        'train_loss': train_loss_results,
        'train_accuracy': train_accuracy_results,
        'test_loss': test_loss_results,
        'test_accuracy': test_accuracy_results
    }
    
    return history

def plot_training_results(histories, model_names, save_path=None):
    """
    Plot training and test accuracy/loss for different models
    
    Args:
        histories: List of training histories
        model_names: List of model names for the legend
        save_path: Optional path to save the figures
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.set_title('Model Accuracy')
    for i, history in enumerate(histories):
        ax1.plot(history['train_accuracy'], label=f'{model_names[i]} Train')
        ax1.plot(history['test_accuracy'], label=f'{model_names[i]} Test')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.set_title('Model Loss')
    for i, history in enumerate(histories):
        ax2.plot(history['train_loss'], label=f'{model_names[i]} Train')
        ax2.plot(history['test_loss'], label=f'{model_names[i]} Test')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def run_experiment(model_types, hidden_units_list, epochs=10, trials=3):
    """
    Run experiments for multiple model types and configurations
    
    Args:
        model_types: List of model types to evaluate
        hidden_units_list: List of hidden units to test
        epochs: Number of training epochs
        trials: Number of trials to run for each configuration
    """
    # Load and preprocess the data
    X_train, y_train, X_test, y_test = load_mnist_data()
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Store results for each configuration
    all_histories = {}
    all_times = {}
    
    for model_type in model_types:
        for units in hidden_units_list:
            config_name = f"{model_type}_{units}units"
            print(f"\n{'-'*50}")
            print(f"Training {config_name}")
            print(f"{'-'*50}")
            
            config_histories = []
            config_times = []
            
            for trial in range(trials):
                print(f"\nTrial {trial + 1}/{trials}")
                
                # Create model
                model = create_model(
                    model_type=model_type,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    hidden_units=units,
                    num_classes=y_train.shape[1]
                )
                
                # Train model and measure time
                start_time = time.time()
                history = train_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    epochs=epochs
                )
                end_time = time.time()
                training_time = end_time - start_time
                
                config_histories.append(history)
                config_times.append(training_time)
                
                print(f"Training time: {training_time:.2f} seconds")
                print(f"Final test accuracy: {history['test_accuracy'][-1]:.4f}")
            
            # Average the histories for this configuration
            avg_history = {
                'train_loss': np.mean([h['train_loss'] for h in config_histories], axis=0),
                'train_accuracy': np.mean([h['train_accuracy'] for h in config_histories], axis=0),
                'test_loss': np.mean([h['test_loss'] for h in config_histories], axis=0),
                'test_accuracy': np.mean([h['test_accuracy'] for h in config_histories], axis=0)
            }
            
            avg_time = np.mean(config_times)
            
            # Store the results
            all_histories[config_name] = avg_history
            all_times[config_name] = avg_time
            
            print(f"\nAverage training time for {config_name}: {avg_time:.2f} seconds")
            print(f"Average final test accuracy: {avg_history['test_accuracy'][-1]:.4f}")
            
            # Plot the training curves for this configuration
            plot_training_results(
                [avg_history],
                [config_name],
                save_path=f"{config_name}_training_curves.png"
            )
    
    # Compare different models
    model_names = list(all_histories.keys())
    histories = [all_histories[name] for name in model_names]
    
    print("\nComparison of models:")
    for name in model_names:
        print(f"{name}: {all_histories[name]['test_accuracy'][-1]:.4f} accuracy, {all_times[name]:.2f} seconds")
    
    # Plot comparison of all models
    plot_training_results(
        histories,
        model_names,
        save_path="model_comparison.png"
    )
    
    return all_histories, all_times

if __name__ == "__main__":
    # Configuration for experiments
    model_types = ['gru', 'mgu']
    hidden_units_list = [128, 256]  # You can try different sizes: 50, 128, 256, 512
    epochs = 10
    trials = 3
    
    # Run the experiments
    all_histories, all_times = run_experiment(
        model_types=model_types,
        hidden_units_list=hidden_units_list,
        epochs=epochs,
        trials=trials
    )
    
    # Print final results
    print("\nFinal Results:")
    print("-" * 50)
    print("Model | Test Accuracy | Training Time (s)")
    print("-" * 50)
    
    for model_name in sorted(all_histories.keys()):
        history = all_histories[model_name]
        time = all_times[model_name]
        print(f"{model_name} | {history['test_accuracy'][-1]:.4f} | {time:.2f}")
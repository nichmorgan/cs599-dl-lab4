"""
Functions for running single model experiments and configurations.
"""

import os
import time
import datetime
import numpy as np
import json

from models import create_model
from training import train_model
from visualization import plot_training_results


def run_single_configuration(model_type, hidden_units, X_train, y_train, X_test, y_test,
                            epochs=10, batch_size=64, learning_rate=0.001, trials=3,
                            log_interval=10, config_dir=None):
    """
    Run a single model configuration with multiple trials
    
    Args:
        model_type: Type of RNN cell to use ('gru', 'mgu', 'lstm')
        hidden_units: Number of hidden units in the model
        X_train, y_train: Training data
        X_test, y_test: Testing data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        trials: Number of trials to run
        log_interval: Number of batches between logging updates
        config_dir: Directory to save results
        
    Returns:
        tuple: (avg_history, avg_time, std_time, avg_error, std_error)
    """
    # Create output directory if not exists
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    
    # Create configuration log file
    if config_dir:
        config_log = os.path.join(config_dir, "configuration_log.txt")
        with open(config_log, 'w') as f:
            f.write(f"Configuration: {model_type} with {hidden_units} hidden units\n")
            f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Trials: {trials}\n")
            f.write("-" * 50 + "\n\n")
    
    # Run multiple trials
    config_histories = []
    config_times = []
    
    for trial in range(trials):
        print(f"\n{'-'*50}")
        print(f"Trial {trial + 1}/{trials} for {model_type} with {hidden_units} hidden units")
        print(f"{'-'*50}")
        
        # Log to configuration file
        if config_dir:
            with open(config_log, 'a') as f:
                f.write(f"\nTrial {trial + 1}/{trials}\n")
                f.write("-" * 30 + "\n")
        
        # Create model
        model = create_model(
            model_type=model_type,
            input_shape=(X_train.shape[1], X_train.shape[2]),
            hidden_units=hidden_units,
            num_classes=y_train.shape[1]
        )
        
        # Create trial directory
        if config_dir:
            trial_dir = os.path.join(config_dir, f"trial_{trial+1}")
            os.makedirs(trial_dir, exist_ok=True)
        else:
            trial_dir = None
        
        # Train model and measure time
        start_time = time.time()
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            log_interval=log_interval,
            output_dir=trial_dir
        )
        end_time = time.time()
        training_time = end_time - start_time
        
        config_histories.append(history)
        config_times.append(training_time)
        
        # Trial summary
        train_loss = history['train_loss'][-1]
        train_accuracy = history['train_accuracy'][-1]
        test_loss = history['test_loss'][-1]
        test_accuracy = history['test_accuracy'][-1]
        test_error = 1.0 - test_accuracy
        
        trial_summary = (
            f"Trial {trial + 1}/{trials} completed in {training_time:.2f} seconds\n"
            f"Final train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}\n"
            f"Final test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}\n"
            f"Final test error: {test_error:.4f}"
        )
        
        print("\n" + trial_summary)
        
        # Log to configuration file
        if config_dir:
            with open(config_log, 'a') as f:
                f.write(trial_summary + "\n")
        
        # Plot and save individual trial results
        if config_dir:
            trial_history = {
                'train_loss': history['train_loss'],
                'train_accuracy': history['train_accuracy'],
                'test_loss': history['test_loss'],
                'test_accuracy': history['test_accuracy']
            }
            
            plot_training_results(
                [trial_history],
                [f"{model_type}_{hidden_units}units (Trial {trial+1})"],
                save_path=os.path.join(trial_dir, f"training_curves.png")
            )
    
    # Average the histories for this configuration
    avg_history = {
        'train_loss': np.mean([h['train_loss'] for h in config_histories], axis=0),
        'train_accuracy': np.mean([h['train_accuracy'] for h in config_histories], axis=0),
        'test_loss': np.mean([h['test_loss'] for h in config_histories], axis=0),
        'test_accuracy': np.mean([h['test_accuracy'] for h in config_histories], axis=0)
    }
    
    avg_time = np.mean(config_times)
    std_time = np.std(config_times)
    
    avg_test_accuracy = avg_history['test_accuracy'][-1]
    avg_test_error = 1.0 - avg_test_accuracy
    std_test_error = np.std([1.0 - h['test_accuracy'][-1] for h in config_histories])
    
    # Configuration summary
    config_summary = (
        f"\nAverage results for {model_type} with {hidden_units} hidden units over {trials} trials:\n"
        f"Training time: {avg_time:.2f} ± {std_time:.2f} seconds\n"
        f"Final train loss: {avg_history['train_loss'][-1]:.4f}\n"
        f"Final train accuracy: {avg_history['train_accuracy'][-1]:.4f}\n"
        f"Final test loss: {avg_history['test_loss'][-1]:.4f}\n"
        f"Final test accuracy: {avg_test_accuracy:.4f}\n"
        f"Final test error: {avg_test_error:.4f} ± {std_test_error:.4f}"
    )
    
    print(config_summary)
    
    # Log to configuration file
    if config_dir:
        with open(config_log, 'a') as f:
            f.write("\n" + config_summary + "\n")
            f.write("-" * 50 + "\n")
        
        # Save average history to file
        avg_history_file = os.path.join(config_dir, "average_history.json")
        with open(avg_history_file, 'w') as f:
            json.dump({k: v.tolist() for k, v in avg_history.items()}, f, indent=4)
        
        # Plot the average training curves for this configuration
        plot_training_results(
            [avg_history],
            [f"{model_type}_{hidden_units}units"],
            save_path=os.path.join(config_dir, f"average_curves.png")
        )
    
    return avg_history, avg_time, std_time, avg_test_error, std_test_error
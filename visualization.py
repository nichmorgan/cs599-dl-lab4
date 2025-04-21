"""
Visualization functions for plotting training results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_results(histories, model_names, save_path=None):
    """
    Plot training and test accuracy/loss for different models
    
    Args:
        histories: List of training histories
        model_names: List of model names for the legend
        save_path: Optional path to save the figures
    """
    print(f"Plotting training results for {', '.join(model_names)}...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training & validation accuracy
    ax1.set_title('Model Accuracy')
    for i, history in enumerate(histories):
        ax1.plot(history['train_accuracy'], label=f'{model_names[i]} Train')
        ax1.plot(history['test_accuracy'], label=f'{model_names[i]} Test')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot training & validation error (1 - accuracy)
    ax2.set_title('Model Error')
    for i, history in enumerate(histories):
        train_error = [1.0 - acc for acc in history['train_accuracy']]
        test_error = [1.0 - acc for acc in history['test_accuracy']]
        ax2.plot(train_error, label=f'{model_names[i]} Train')
        ax2.plot(test_error, label=f'{model_names[i]} Test')
    ax2.set_ylabel('Error Rate')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Plot training & validation loss
    ax3.set_title('Model Loss')
    for i, history in enumerate(histories):
        ax3.plot(history['train_loss'], label=f'{model_names[i]} Train')
        ax3.plot(history['test_loss'], label=f'{model_names[i]} Test')
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Epoch')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)  # Close the figure to free memory


def plot_comparison_bars(results, save_path=None):
    """
    Create bar charts comparing different models
    
    Args:
        results: Dictionary with model names as keys and (accuracy, error, time) tuples as values
        save_path: Optional path to save the figure
    """
    model_names = list(results.keys())
    accuracies = [results[name][0] for name in model_names]
    errors = [results[name][1] for name in model_names]
    times = [results[name][2] for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracy/error bars
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Accuracy')
    ax1.bar(x + width/2, errors, width, label='Error')
    ax1.set_ylabel('Rate')
    ax1.set_title('Model Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(True, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(accuracies):
        ax1.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
    for i, v in enumerate(errors):
        ax1.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
    
    # Plot training time bars
    ax2.bar(model_names, times, color='green')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Time')
    ax2.grid(True, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(times):
        ax2.text(i, v + 0.5, f"{v:.1f}s", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.close(fig)  # Close the figure to free memory
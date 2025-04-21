"""
Visualization functions for plotting training results with minimal dependencies.
Uses matplotlib-base only to avoid GLIBCXX version conflicts on clusters.
"""

import os
import sys
import numpy as np

# Import matplotlib with Agg backend (no GUI required)
try:
    import matplotlib
    matplotlib.use('Agg')  # Set before importing pyplot
    from matplotlib import pyplot as plt
except ImportError as e:
    print("ERROR: matplotlib-base is required for visualization.")
    print(f"Import error: {e}")
    print("Please install matplotlib-base:")
    print("  conda install -c conda-forge matplotlib-base")
    sys.exit(1)  # Exit with error code

def plot_training_results(histories, model_names, save_path=None):
    """
    Plot training and test accuracy/loss for different models
    
    Args:
        histories: List of training histories
        model_names: List of model names for the legend
        save_path: Optional path to save the figures
    """
    print(f"Plotting training results for {', '.join(model_names)}...")
    
    # Create three separate plots for better readability
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training & validation accuracy
    ax1 = axes[0]
    for i, history in enumerate(histories):
        epochs = range(1, len(history['train_accuracy']) + 1)
        ax1.plot(epochs, history['train_accuracy'], 'o-', label=f'{model_names[i]} Train')
        ax1.plot(epochs, history['test_accuracy'], 's--', label=f'{model_names[i]} Test')
    
    ax1.set_title('Model Accuracy', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot training & validation error (1 - accuracy)
    ax2 = axes[1]
    for i, history in enumerate(histories):
        epochs = range(1, len(history['train_accuracy']) + 1)
        train_error = [1.0 - acc for acc in history['train_accuracy']]
        test_error = [1.0 - acc for acc in history['test_accuracy']]
        ax2.plot(epochs, train_error, 'o-', label=f'{model_names[i]} Train')
        ax2.plot(epochs, test_error, 's--', label=f'{model_names[i]} Test')
    
    ax2.set_title('Model Error', fontweight='bold')
    ax2.set_ylabel('Error Rate')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Plot training & validation loss
    ax3 = axes[2]
    for i, history in enumerate(histories):
        epochs = range(1, len(history['train_loss']) + 1)
        ax3.plot(epochs, history['train_loss'], 'o-', label=f'{model_names[i]} Train')
        ax3.plot(epochs, history['test_loss'], 's--', label=f'{model_names[i]} Test')
    
    ax3.set_title('Model Loss', fontweight='bold')
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
    
    # Plot accuracy bars
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Accuracy')
    ax1.bar(x + width/2, errors, width, label='Error')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Rate')
    ax1.set_title('Model Performance', fontweight='bold')
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
    ax2.bar(x, times, color="green")
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.grid(True, axis='y')
    
    # Add value labels on time bars
    for i, v in enumerate(times):
        ax2.text(i, v + 0.5, f"{v:.1f}s", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.close(fig)  # Close the figure to free memory
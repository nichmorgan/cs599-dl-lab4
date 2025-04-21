"""
Visualization functions for plotting training results with seaborn.
"""

import os
import sys
import numpy as np

# Try to import seaborn with minimal matplotlib dependencies
try:
    import seaborn as sns
    from matplotlib import pyplot as plt
    # Force the Agg backend which doesn't require X11
    import matplotlib
    matplotlib.use('Agg')
    # Set seaborn styling
    sns.set(style="whitegrid", context="paper", font_scale=1.2)
except ImportError as e:
    print("ERROR: Required visualization libraries are not available.")
    print(f"Import error: {e}")
    print("Visualization capabilities are essential for this experiment.")
    print("Please install the required packages:")
    print("  pip install seaborn")
    print("Terminating execution.")
    sys.exit(1)  # Exit with error code

def plot_training_results(histories, model_names, save_path=None):
    """
    Plot training and test accuracy/loss for different models using seaborn
    
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
        sns.lineplot(x=epochs, y=history['train_accuracy'], label=f'{model_names[i]} Train', ax=ax1)
        sns.lineplot(x=epochs, y=history['test_accuracy'], label=f'{model_names[i]} Test', ax=ax1, linestyle='--')
    
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
        sns.lineplot(x=epochs, y=train_error, label=f'{model_names[i]} Train', ax=ax2)
        sns.lineplot(x=epochs, y=test_error, label=f'{model_names[i]} Test', ax=ax2, linestyle='--')
    
    ax2.set_title('Model Error', fontweight='bold')
    ax2.set_ylabel('Error Rate')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Plot training & validation loss
    ax3 = axes[2]
    for i, history in enumerate(histories):
        epochs = range(1, len(history['train_loss']) + 1)
        sns.lineplot(x=epochs, y=history['train_loss'], label=f'{model_names[i]} Train', ax=ax3)
        sns.lineplot(x=epochs, y=history['test_loss'], label=f'{model_names[i]} Test', ax=ax3, linestyle='--')
    
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
    Create bar charts comparing different models using seaborn
    
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
    performance_data = []
    for i, name in enumerate(model_names):
        performance_data.append({"Model": name, "Metric": "Accuracy", "Value": accuracies[i]})
        performance_data.append({"Model": name, "Metric": "Error", "Value": errors[i]})
    
    # Convert to long-form data for seaborn
    performance_df = {"Model": [], "Metric": [], "Value": []}
    for item in performance_data:
        performance_df["Model"].append(item["Model"])
        performance_df["Metric"].append(item["Metric"])
        performance_df["Value"].append(item["Value"])
    
    # Plot with seaborn
    sns.barplot(x="Model", y="Value", hue="Metric", data=performance_df, ax=ax1)
    ax1.set_ylabel('Rate')
    ax1.set_title('Model Performance', fontweight='bold')
    ax1.grid(True, axis='y')
    
    # Add value labels on bars
    bars = ax1.patches
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f"{height:.3f}",
            ha="center", va="bottom"
        )
    
    # Plot training time bars
    time_data = {"Model": model_names, "Time": times}
    sns.barplot(x="Model", y="Time", data=time_data, color="green", ax=ax2)
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time', fontweight='bold')
    ax2.grid(True, axis='y')
    
    # Add value labels on time bars
    for i, v in enumerate(times):
        ax2.text(i, v + 0.5, f"{v:.1f}s", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.close(fig)  # Close the figure to free memory
#!/usr/bin/env python3
"""
Main entry point for RNN cell experiments with Dask integration.
Run different configurations on the notMNIST dataset using either local or SLURM resources.
"""

import argparse
import os
import sys
import datetime
import json
import time
import itertools
from pathlib import Path

import dask
from dask.distributed import Client, LocalCluster, wait
from dask_jobqueue import SLURMCluster
import numpy as np
import tensorflow as tf

# Disable progress bars to avoid Qt errors and other display issues
dask.config.set({"distributed.diagnostics.progress": False})

from data_loader import load_notmnist_data, preprocess_data
from models import create_model
from training import train_model
from visualization import plot_training_results, plot_comparison_bars
from experiment import run_single_configuration


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RNN Cell Comparison Experiments with Dask")
    
    # Experiment configuration
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name for the experiment (default: timestamp)")
    parser.add_argument("--output-dir", type=str, default="experiments",
                        help="Directory to save experiment results (default: experiments)")
    
    # Model configuration
    parser.add_argument("--model-types", type=str, nargs="+", default=["gru", "mgu"],
                        choices=["gru", "mgu", "lstm"], 
                        help="Model types to evaluate (default: gru mgu)")
    parser.add_argument("--hidden-units", type=int, nargs="+", default=[128, 256],
                        help="List of hidden unit sizes to test (default: 128 256)")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer (default: 0.001)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials to run for each configuration (default: 3)")
    
    # Execution options
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Number of batches between logging updates (default: 10)")
    
    # Dask configuration
    parser.add_argument("--scheduler", type=str, default="local", choices=["local", "slurm"],
                        help="Dask scheduler to use (default: local)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers for local scheduler (default: 4)")
    parser.add_argument("--slurm-cpus", type=int, default=8,
                        help="CPUs per SLURM task (default: 8)")
    parser.add_argument("--slurm-mem", type=str, default="16GB",
                        help="Memory per SLURM task (default: 16GB)")
    parser.add_argument("--slurm-time", type=str, default="08:00:00",
                        help="Time limit per SLURM task (default: 08:00:00)")
    
    return parser.parse_args()


def setup_dask_client(args):
    """Set up Dask client based on scheduler choice."""
    if args.scheduler == "local":
        print(f"Setting up local Dask cluster with {args.workers} workers...")
        cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
        client = Client(cluster)
    else:  # slurm
        print(f"Setting up SLURM Dask cluster...")
        cluster = SLURMCluster(
            cores=args.slurm_cpus,
            memory=args.slurm_mem,
            walltime=args.slurm_time,
            log_directory="dask_logs",
            local_directory="dask_local",
            job_extra_directives=[
                "--output=dask_logs/slurm-%j.out", 
                "--error=dask_logs/slurm-%j.err"
            ],
        )
        
        # Scale the cluster - one worker per configuration
        n_configs = len(args.model_types) * len(args.hidden_units)
        cluster.scale(n_configs)
        
        # Connect to the cluster
        client = Client(cluster)
        
        # Wait for workers to start
        print("Waiting for SLURM workers to start...")
        client.wait_for_workers(n_workers=1)  # Wait for at least one worker
    
    print(f"Dask dashboard available at: {client.dashboard_link}")
    return client


@dask.delayed
def run_experiment_configuration(model_type, hidden_units, epochs, batch_size, learning_rate, trials, log_interval, config_dir):
    """
    Run a single experiment configuration, wrapped for Dask.
    
    This function needs to be self-contained since it will run on a worker node.
    """
    # Set deterministic behavior for reproducibility
    tf.random.set_seed(0)
    np.random.seed(0)
    
    print(f"Starting experiment configuration: {model_type} with {hidden_units} hidden units")
    
    # Create configuration directory
    os.makedirs(config_dir, exist_ok=True)
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_notmnist_data()
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Run the configuration
    avg_history, avg_time, std_time, avg_error, std_error = run_single_configuration(
        model_type=model_type,
        hidden_units=hidden_units,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        trials=trials,
        log_interval=log_interval,
        config_dir=config_dir
    )
    
    print(f"Completed experiment configuration: {model_type} with {hidden_units} hidden units")
    
    # Return a summary of the results
    return {
        'model_type': model_type,
        'hidden_units': hidden_units,
        'avg_history': avg_history,
        'avg_time': avg_time,
        'std_time': std_time,
        'avg_error': avg_error,
        'std_error': std_error,
        'config_dir': config_dir
    }


def track_task_progress(futures):
    """Display simple progress for tasks without using Qt or fancy progress bars."""
    total = len(futures)
    last_completed = 0
    
    while True:
        done = sum(f.done() for f in futures)
        if done > last_completed:
            last_completed = done
            completion_pct = done / total * 100
            print(f"Progress: {done}/{total} configurations completed ({completion_pct:.1f}%)")
        
        if done == total:
            break
            
        time.sleep(2)  # Update every 2 seconds


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create experiment directory
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create log directories
    os.makedirs("dask_logs", exist_ok=True)
    os.makedirs("dask_local", exist_ok=True)
    
    # Save experiment configuration
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"Experiment: {args.experiment_name}")
    print(f"Results will be saved to: {experiment_dir}")
    print(f"Configuration saved to: {config_path}")
    
    # Set up Dask client
    client = setup_dask_client(args)
    print(f"Connected to Dask cluster with {len(client.scheduler_info()['workers'])} workers")
    
    # Create all possible configurations
    configurations = list(itertools.product(args.model_types, args.hidden_units))
    print(f"Running {len(configurations)} configurations, each with {args.trials} trials")
    
    # Generate Dask tasks for each configuration
    tasks = []
    for model_type, hidden_units in configurations:
        config_name = f"{model_type}_{hidden_units}units"
        config_dir = os.path.join(experiment_dir, config_name)
        
        task = run_experiment_configuration(
            model_type=model_type,
            hidden_units=hidden_units,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            trials=args.trials,
            log_interval=args.log_interval,
            config_dir=config_dir
        )
        tasks.append(task)
    
    # Execute all tasks
    print(f"Submitting {len(tasks)} tasks to Dask cluster...")
    futures = client.compute(tasks)
    
    # Show simple progress tracking without fancy progress bars
    print("\nTracking progress (this may take a while):")
    
    # Create a separate thread to track progress
    from threading import Thread
    progress_thread = Thread(target=track_task_progress, args=(futures,), daemon=True)
    progress_thread.start()
    
    # Wait for completion
    print("\nWaiting for all tasks to complete...")
    results = client.gather(futures)
    
    # Combine results for comparison
    all_histories = {}
    all_times = {}
    all_errors = {}
    comparison_results = {}
    
    for result in results:
        model_type = result['model_type']
        hidden_units = result['hidden_units']
        config_name = f"{model_type}_{hidden_units}units"
        
        all_histories[config_name] = result['avg_history']
        all_times[config_name] = result['avg_time']
        all_errors[config_name] = (result['avg_error'], result['std_error'])
        
        # For comparison plots
        avg_test_accuracy = 1.0 - result['avg_error']
        comparison_results[config_name] = (avg_test_accuracy, result['avg_error'], result['avg_time'])
    
    # Compare different models
    model_names = list(all_histories.keys())
    histories = [all_histories[name] for name in model_names]
    
    # Create summary report
    report_path = os.path.join(experiment_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Comparison of all models:\n")
        f.write("-" * 70 + "\n")
        f.write("Model | Test Error | Test Accuracy | Training Time (s)\n")
        f.write("-" * 70 + "\n")
        
        for name in model_names:
            test_accuracy = 1.0 - all_errors[name][0]
            test_error, error_std = all_errors[name]
            f.write(f"{name} | {test_error:.4f} ± {error_std:.4f} | {test_accuracy:.4f} | {all_times[name]:.2f}\n")
    
    print(f"Summary report saved to: {report_path}")
    
    # Print results table to console
    print("\nResults Summary:")
    print("-" * 70)
    print("Model | Test Error | Test Accuracy | Training Time (s)")
    print("-" * 70)
    for name in sorted(model_names):
        test_accuracy = 1.0 - all_errors[name][0]
        test_error, error_std = all_errors[name]
        print(f"{name} | {test_error:.4f} ± {error_std:.4f} | {test_accuracy:.4f} | {all_times[name]:.2f}")
    
    # Plot comparison of all models' training curves
    print("\nCreating comparison plots...")
    plot_training_results(
        histories,
        model_names,
        save_path=os.path.join(experiment_dir, "model_comparison_curves.png")
    )
    
    # Plot comparison bar charts
    plot_comparison_bars(
        comparison_results,
        save_path=os.path.join(experiment_dir, "model_comparison_bars.png")
    )
    
    print(f"\nExperiment completed successfully!")
    print(f"All results are available in: {experiment_dir}")
    
    # Close Dask client
    client.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
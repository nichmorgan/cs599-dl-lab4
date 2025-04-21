"""
Training functions for RNN models with improved progress reporting.
"""

import os
import time
import datetime
import tensorflow as tf
import numpy as np
import json
import sys
from tqdm.auto import tqdm


class TrainingLogger:
    """Custom logger for training progress that works with tqdm progress bars."""
    
    def __init__(self, log_file=None, log_interval=10, total_batches=None):
        self.log_file = log_file
        self.log_interval = log_interval
        self.total_batches = total_batches
        self.progress_bar = None
        
        # Create log file if needed
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            os.makedirs(log_dir, exist_ok=True)
    
    def start_epoch(self, epoch, total_epochs):
        """Start tracking a new epoch."""
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.batch_times = []
        
        # Create progress bar for this epoch
        if self.total_batches:
            self.progress_bar = tqdm(
                total=self.total_batches,
                desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
                leave=True,
                dynamic_ncols=True,
                unit="batch"
            )
        
        # Log epoch start
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1}/{total_epochs}\n")
                f.write("-" * 30 + "\n")
    
    def update_batch(self, batch_idx, loss, accuracy, batch_time=None):
        """Update progress for a batch."""
        if batch_time:
            self.batch_times.append(batch_time)
        
        # Only log at specified intervals
        do_log = ((batch_idx + 1) % self.log_interval == 0) or ((batch_idx + 1) == self.total_batches)
        
        if self.progress_bar:
            # Update progress bar
            self.progress_bar.update(1)
            
            # Update progress bar description at log intervals
            if do_log:
                avg_time = np.mean(self.batch_times[-self.log_interval:])
                self.progress_bar.set_postfix({
                    'Loss': f"{loss:.4f}",
                    'Acc': f"{accuracy:.4f}",
                    'Time/batch': f"{avg_time:.2f}s"
                })
        
        # Log to file if needed
        if do_log and self.log_file:
            avg_time = np.mean(self.batch_times[-self.log_interval:])
            progress_msg = (
                f"Batch {batch_idx+1}/{self.total_batches} "
                f"[{(batch_idx+1)/self.total_batches*100:.1f}%] - "
                f"Time: {avg_time:.2f}s - "
                f"Loss: {loss:.4f} - "
                f"Accuracy: {accuracy:.4f}"
            )
            
            with open(self.log_file, 'a') as f:
                f.write(f"{progress_msg}\n")
    
    def complete_epoch(self, train_metrics, test_metrics, epoch_time, test_time):
        """Finalize an epoch with summary metrics."""
        # Close progress bar
        if self.progress_bar:
            self.progress_bar.close()
        
        # Print test evaluation progress
        test_bar = tqdm(
            total=100, 
            desc=f"Epoch {self.epoch+1}/{self.total_epochs} [Test]", 
            leave=True,
            dynamic_ncols=True,
            unit="%"
        )
        test_bar.update(100)  # Complete immediately
        test_bar.set_postfix({
            'Loss': f"{test_metrics['loss']:.4f}", 
            'Acc': f"{test_metrics['accuracy']:.4f}", 
            'Error': f"{1.0-test_metrics['accuracy']:.4f}"
        })
        test_bar.close()
        
        # Create epoch summary
        epoch_summary = (
            f"Epoch {self.epoch+1}/{self.total_epochs} completed in {epoch_time:.2f}s (test: {test_time:.2f}s)\n"
            f"Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}\n"
            f"Test Loss: {test_metrics['loss']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}\n"
            f"Test Error: {1.0 - test_metrics['accuracy']:.4f}"
        )
        
        # Print summary with spacing for readability
        print("\n" + epoch_summary)
        
        # Log to file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write("\n" + epoch_summary + "\n")
                f.write("-" * 80 + "\n")


def train_model(model, X_train, y_train, X_test, y_test, 
                batch_size=64, epochs=10, learning_rate=0.001,
                log_interval=10, output_dir=None):
    """
    Train the RNN model with improved progress reporting
    
    Args:
        model: RNN model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        log_interval: Number of batches between logging updates
        output_dir: Directory to save training logs
        
    Returns:
        history: Training history
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    print(f"Creating datasets with batch size {batch_size}...")
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
    # Calculate total batches per epoch for logging
    total_train_batches = len(list(train_dataset))
    
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    
    # Setup logging
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"training_log.txt")
        metrics_file = os.path.join(output_dir, "metrics.jsonl")
    else:
        log_dir = "training_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        metrics_file = os.path.join(os.path.dirname(log_file), "metrics.jsonl")
    
    # Write initial log entry
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Hidden units: {model.hidden_units}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Total epochs: {epochs}\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Total training batches per epoch: {total_train_batches}\n")
        f.write("=" * 80 + "\n\n")
    
    print(f"Training for {epochs} epochs...")
    print(f"Training log saved to: {log_file}")
    
    # Create logger
    logger = TrainingLogger(
        log_file=log_file,
        log_interval=log_interval,
        total_batches=total_train_batches
    )
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training loop
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        # Start epoch tracking
        logger.start_epoch(epoch, epochs)
        
        # Track batch progress
        for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
            batch_start_time = time.time()
            
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
            
            # Log progress
            batch_time = time.time() - batch_start_time
            logger.update_batch(
                batch_idx,
                epoch_loss_avg.result().numpy(),
                epoch_accuracy.result().numpy(),
                batch_time
            )
        
        # End of epoch - collect training metrics
        train_loss = epoch_loss_avg.result().numpy()
        train_accuracy = epoch_accuracy.result().numpy()
        train_loss_results.append(train_loss)
        train_accuracy_results.append(train_accuracy)
        
        # Evaluation loop
        test_start_time = time.time()
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
        test_loss = test_loss_avg.result().numpy()
        test_accuracy = test_accuracy.result().numpy()
        test_loss_results.append(test_loss)
        test_accuracy_results.append(test_accuracy)
        
        # Calculate test error
        test_error = 1.0 - test_accuracy
        
        # Calculate times
        epoch_time = time.time() - epoch_start_time
        test_time = time.time() - test_start_time
        
        # Complete epoch logging
        logger.complete_epoch(
            train_metrics={'loss': train_loss, 'accuracy': train_accuracy},
            test_metrics={'loss': test_loss, 'accuracy': test_accuracy},
            epoch_time=epoch_time,
            test_time=test_time
        )
        
        # Write metrics to JSONL for easy parsing
        metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "test_error": float(test_error),
            "epoch_time": float(epoch_time)
        }
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + "\n")
    
    # Training complete
    training_complete_msg = (
        f"\nTraining completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Final test accuracy: {test_accuracy:.4f}\n"
        f"Final test error: {test_error:.4f}"
    )
    
    print(training_complete_msg)
    
    # Log final summary to file
    with open(log_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(training_complete_msg + "\n")
    
    # Save final model weights if output_dir is provided
    if output_dir:
        weights_dir = os.path.join(output_dir, "model_weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        # Save weights as NumPy arrays
        for i, weight in enumerate(model.trainable_variables):
            weight_path = os.path.join(weights_dir, f"weight_{i}.npy")
            np.save(weight_path, weight.numpy())
        
        print(f"Model weights saved to: {weights_dir}")
    
    # Return history for plotting
    history = {
        'train_loss': train_loss_results,
        'train_accuracy': train_accuracy_results,
        'test_loss': test_loss_results,
        'test_accuracy': test_accuracy_results
    }
    
    return history
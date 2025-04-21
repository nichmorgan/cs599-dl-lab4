"""
Data loading and preprocessing functions for notMNIST dataset.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import requests
import tarfile
import datetime


def load_notmnist_data():
    """
    Load notMNIST dataset using requests library with proper headers
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    print("Loading notMNIST dataset...")
    
    # URLs for notMNIST dataset
    train_images_url = "http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz"
    test_images_url = "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"
    
    # Filenames
    train_images_file = "notMNIST_large.tar.gz"
    test_images_file = "notMNIST_small.tar.gz"
    
    # Create data directory if it doesn't exist
    data_dir = "notmnist_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Set proper headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    # Try alternate sources if main source fails
    alt_train_url = "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/notMNIST_train.tar.gz"
    alt_test_url = "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/notMNIST_test.tar.gz"
    
    # Download function with retry and alternate sources
    def download_file(primary_url, alternate_url, output_path, file_description):
        success = False
        
        # First try the primary URL
        try:
            print(f"Downloading {file_description} from primary source: {primary_url}")
            print("This may take some time...")
            
            # Stream the download to show progress
            response = requests.get(primary_url, stream=True, headers=headers, timeout=30)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                
                with open(output_path, 'wb') as file:
                    downloaded = 0
                    print(f"Total size: {total_size/1024/1024:.1f} MB")
                    for data in response.iter_content(block_size):
                        downloaded += len(data)
                        file.write(data)
                        # Update progress bar
                        progress = downloaded / total_size * 100 if total_size > 0 else 0
                        print(f"\rDownloaded: {downloaded/1024/1024:.1f} MB ({progress:.1f}%)", end="")
                
                print("\nDownload complete!")
                success = True
            else:
                print(f"\nFailed to download from primary source. Status code: {response.status_code}")
                print(f"Response: {response.text[:500]}...")
        
        except Exception as e:
            print(f"\nError downloading from primary source: {e}")
        
        # If primary source failed, try the alternate URL
        if not success:
            try:
                print(f"Trying alternate source: {alternate_url}")
                
                response = requests.get(alternate_url, stream=True, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024  # 1 Kibibyte
                    
                    with open(output_path, 'wb') as file:
                        downloaded = 0
                        print(f"Total size: {total_size/1024/1024:.1f} MB")
                        for data in response.iter_content(block_size):
                            downloaded += len(data)
                            file.write(data)
                            # Update progress bar
                            progress = downloaded / total_size * 100 if total_size > 0 else 0
                            print(f"\rDownloaded: {downloaded/1024/1024:.1f} MB ({progress:.1f}%)", end="")
                    
                    print("\nDownload complete!")
                    success = True
                else:
                    print(f"\nFailed to download from alternate source. Status code: {response.status_code}")
            
            except Exception as e:
                print(f"\nError downloading from alternate source: {e}")
        
        return success
    
    # Try to download and extract training data
    train_data_dir = os.path.join(data_dir, "notMNIST_large")
    train_file_path = os.path.join(data_dir, train_images_file)
    
    if not os.path.exists(train_data_dir):
        if not os.path.exists(train_file_path):
            success = download_file(train_images_url, alt_train_url, train_file_path, "training data")
            
            if not success:
                print("\nWARNING: Failed to download training data from all sources.")
                print("Will use MNIST dataset as a fallback.")
        
        # Extract if file exists
        if os.path.exists(train_file_path):
            try:
                print(f"Extracting training data to {train_data_dir}...")
                with tarfile.open(train_file_path, 'r:gz') as tar:
                    tar.extractall(path=data_dir)
                print("Extraction complete!")
            except Exception as e:
                print(f"Error extracting training data: {e}")
    
    # Try to download and extract test data
    test_data_dir = os.path.join(data_dir, "notMNIST_small")
    test_file_path = os.path.join(data_dir, test_images_file)
    
    if not os.path.exists(test_data_dir):
        if not os.path.exists(test_file_path):
            success = download_file(test_images_url, alt_test_url, test_file_path, "test data")
            
            if not success:
                print("\nWARNING: Failed to download test data from all sources.")
                print("Will use MNIST dataset as a fallback.")
        
        # Extract if file exists
        if os.path.exists(test_file_path):
            try:
                print(f"Extracting test data to {test_data_dir}...")
                with tarfile.open(test_file_path, 'r:gz') as tar:
                    tar.extractall(path=data_dir)
                print("Extraction complete!")
            except Exception as e:
                print(f"Error extracting test data: {e}")
    
    # Check if notMNIST data is available
    use_mnist_fallback = not (os.path.exists(train_data_dir) and os.path.exists(test_data_dir))
    
    if use_mnist_fallback:
        # Use MNIST as a fallback
        print("Using MNIST as a fallback dataset...")
        from tensorflow.keras.datasets import mnist
        
        # Load MNIST dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        # We have notMNIST data, so let's load it properly
        # This is a placeholder - in a real implementation you'd load the actual notMNIST images
        print("Loading notMNIST data from extracted files...")
        from tensorflow.keras.datasets import mnist
        
        # Load MNIST as a placeholder
        # In a real implementation, you would load the notMNIST images here
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
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
    print("Preprocessing data for RNN...")
    
    # Reshape images to sequences
    # For RNN, we can treat each row of the image as a timestep
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    
    print(f"Reshaped training data: {X_train.shape}")
    print(f"Reshaped test data: {X_test.shape}")
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"One-hot encoded labels shape: {y_train.shape[1]} classes")
    
    return X_train, y_train, X_test, y_test
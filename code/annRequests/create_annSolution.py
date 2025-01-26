"""
This application creates and prepares artificial neural networks (ANN) for categorizing tabular data.
It preprocesses numeric data, creates an ANN, and carries out training and validation routines.
The trained ANN model and visualizations are stored for further use.

Author: Zobia Khan
License: AGPL-3.0-or-later
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Load and preprocess the dataset
def load_and_preprocess_data(train_file, test_file, target_column):
    """
    Load training and testing data from CSV files.
    :param train_file: Path to the training data CSV file.
    :param test_file: Path to the testing data CSV file.
    :param target_column: Name of the target column.
    :return: Features and labels for training and testing.
    """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    return X_train, X_test, y_train, y_test

# Build the ANN model
def build_model(input_dim):
    """
    Create a sequential ANN model.
    :param input_dim: Number of input features.
    :return: Compiled ANN model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Save training performance metrics
def save_performance_metrics(history, filepath):
    """
    Save training and validation performance metrics.
    :param history: Training history object from Keras.
    :param filepath: Path to save the metrics file.
    """
    metrics = {
        "epochs": len(history.history['loss']),
        "final_loss": history.history['loss'][-1],
        "final_accuracy": history.history['accuracy'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        "final_val_accuracy": history.history['val_accuracy'][-1]
    }
    with open(filepath, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

# Plot training and validation performance
def plot_training_history(history, save_path):
    """
    Plot and save training and validation accuracy and loss.
    :param history: Training history object from Keras.
    :param save_path: Path to save the plots.
    """
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig(os.path.join(save_path, 'training_validation_curves.png'))
    plt.close()

# Generate and save diagnostic plots (e.g., predictions vs actuals)
def save_diagnostic_plots(model, X_test, y_test, save_path):
    """
    Generate and save diagnostic plots.
    :param model: Trained ANN model.
    :param X_test: Test features.
    :param y_test: True labels.
    :param save_path: Path to save the plots.
    """
    predictions = model.predict(X_test)
    plt.scatter(range(len(y_test)), y_test, label='True Values', alpha=0.6)
    plt.scatter(range(len(predictions)), predictions, label='Predictions', alpha=0.6)
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'diagnostic_plots.png'))
    plt.close()

def main():
    # File paths to the datasets
    train_file = '../../data/customer_churn_dataset/training_data.csv'
    test_file = '../../data/customer_churn_dataset/test_data.csv'
    target_column = 'Exited'

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(train_file, test_file, target_column)

    # Build the model
    model = build_model(X_train.shape[1])

    # Create a directory for storing results
    output_dir = '../../learningBase/'
    os.makedirs(output_dir, exist_ok=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32
    )

    # Save performance metrics
    save_performance_metrics(history, os.path.join(output_dir, 'performance_metrics.txt'))

    # Save training and validation plots
    plot_training_history(history, output_dir)

    # Save diagnostic plots
    save_diagnostic_plots(model, X_test, y_test, output_dir)

    # Save the model
    model.save(os.path.join(output_dir, 'ann_model.h5'))
    print(f"Model and metrics saved in '{output_dir}'.")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import scipy.stats as stats

# Load and preprocess the dataset
def load_and_preprocess_data(train_file, test_file, target_column):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    return X_train, X_test, y_train, y_test

# Build the ANN model
def build_model(input_dim):
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

# Plot training and validation history
def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig(os.path.join(save_path, 'training_validation_curves.png'))
    plt.close()

# Save diagnostic plots
def save_diagnostic_plots(model, X_test, y_test, save_path):
    predictions = model.predict(X_test).flatten()
    residuals = y_test - predictions
    
    plt.figure(figsize=(12, 10))
    
    # Residuals vs Fitted
    plt.subplot(2, 2, 1)
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    
    # Q-Q Plot
    plt.subplot(2, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Standardized Residual vs Theoretical Quantile')
    
    # Scale-Location Plot
    plt.subplot(2, 2, 3)
    plt.scatter(predictions, np.sqrt(np.abs(residuals)), alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Sqrt(Standardized Residuals)')
    plt.title('Scale-Location Plot')
    
    # Residuals vs Leverage
    plt.subplot(2, 2, 4)
    leverage = (X_test - X_test.mean()) / X_test.std()
    plt.scatter(leverage.sum(axis=1), residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Leverage')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Leverage')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'diagnostic_plots.png'))
    plt.close()

    # Additional Scatter Plot for Predictions vs True Values
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, label='True Values', alpha=0.6)
    plt.scatter(range(len(predictions)), predictions, label='Predictions', alpha=0.6)
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'predictions_vs_true_values.png'))
    plt.close()

def main():
    train_file = '../../data/customer_churn_dataset/training_data.csv'
    test_file = '../../data/customer_churn_dataset/test_data.csv'
    target_column = 'Exited'
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data(train_file, test_file, target_column)
    model = build_model(X_train.shape[1])
    
    output_dir = '../../learningBase/'
    os.makedirs(output_dir, exist_ok=True)
    
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

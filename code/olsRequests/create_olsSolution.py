"""
This application creates and prepares OLS Model for categorizing tabular data.
It preprocesses numeric data, creates OLS Model, and carries out training and validation routines.
The OLS model and visualizations are stored for further use.

Author: Unaiza Afridi
License: AGPL-3.0-or-later
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic
import scipy.stats as stats

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

def train_ols_model(X_train, y_train):
    """
    Train and fit an OLS regression model.
    """
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    return model

def evaluate_ols_model(model, X, y, data_type):
    """
    Evaluate the OLS model and return performance metrics.
    """
    X = sm.add_constant(X)
    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    print(f"{data_type} Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")

    return predictions, rmse, r2

def diagnostic_plots_ols(model, path):
    """
    Generate and save diagnostic plots (Residuals and QQ Plot).
    """
   
    # Create diagnostic plots
    diagnostic = LinearRegDiagnostic(model)
    diagnostic(plot_context='seaborn-v0_8-paper')
    plt.savefig(os.path.join(path, 'diagnostic_plots_ols.png'))
    plt.close()

def scatter_plot_ols(y_test, test_predictions, path):
    """
    Generate and save scatter plot of OLS predictions vs actual values.
    :param y_test: Test labels.
    :param test_predictions: test predicted values.
    :param path: Path to save the plots.
    """
    plt.figure(figsize=(6, 6))
    
    plt.scatter(range(len(y_test)), y_test, label='True Values', alpha=0.6)
    plt.scatter(range(len(test_predictions)), test_predictions, label='Predictions', alpha=0.6)
    
    # plt.scatter(y_test, test_predictions, alpha=0.5, label='OLS Predictions')
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('OLS Predictions vs Actual Values')
    plt.legend()
    plt.savefig(os.path.join(path, 'scatter_plot_ols.png'))
    plt.close()

def save_diagnostic_plots(model, X_test, y_test, save_path):
    X = sm.add_constant(X_test)
    predictions = model.predict(X)
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
    plt.savefig(os.path.join(save_path, 'diagnostic_plots_ols.png'))
    plt.close()


def main():
    # File paths to the datasets
    train_file = '../../data/customer_churn_dataset/training_data.csv'
    test_file = '../../data/customer_churn_dataset/test_data.csv'
    target_column = 'Exited'

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(train_file, test_file, target_column)

    # Create a directory for storing results
    output_dir = '../../learningBaseOLS/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the OLS Model
    ols_model = train_ols_model(X_train, y_train)

    # Evaluate OLS Model on training and testing Data
    train_predictions, train_rmse, train_r2 = evaluate_ols_model(ols_model, X_train, y_train, "Training Data")
    test_predictions, test_rmse, test_r2 = evaluate_ols_model(ols_model, X_test, y_test, "Testing Data")
    
    # Save training and validation performance metrics
    with open(os.path.join(output_dir, 'performance_metrics_ols.txt'), 'w') as f:
        f.write(f"Training Data RMSE: {train_rmse:.4f}\n")
        f.write(f"Training Data R²: {train_r2:.4f}\n")
        f.write(f"Validation Data RMSE: {test_rmse:.4f}\n")
        f.write(f"Validation Data R²: {test_r2:.4f}\n")

    # Generate and Save Diagnostic and Scatter Plots
    # diagnostic_plots_ols(ols_model, output_dir)
    scatter_plot_ols(y_test, test_predictions, output_dir)
    save_diagnostic_plots(ols_model, X_test, y_test, output_dir)

    ols_model.save(os.path.join(output_dir, 'ols_model.pkl'))
    print(f"OLS Model and metrics saved in '{output_dir}'.")

if __name__ == "__main__":
    main()

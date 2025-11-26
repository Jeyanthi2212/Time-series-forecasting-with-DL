"""
Configuration file for Time Series Forecasting with XAI
Contains all hyperparameters and settings
"""

import os

# Directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data generation parameters
DATA_CONFIG = {
    'n_samples': 5000,
    'n_features': 3,  # Number of input features (excluding target)
    'noise_level': 0.1,
    'trend_strength': 0.02,
    'seasonal_period': 50,
    'seasonal_amplitude': 2.0,
    'random_seed': 42
}

# Model architecture parameters
MODEL_CONFIG = {
    'lookback_window': 30,  # Number of past timesteps to use
    'forecast_horizon': 10,  # Number of future steps to predict
    'lstm_units': [128, 64],  # Units in each LSTM layer
    'dropout_rate': 0.2,
    'attention': True,  # Use attention mechanism
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping_patience': 15
}

# Data split ratios
SPLIT_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15
}

# XAI parameters
XAI_CONFIG = {
    'n_background_samples': 100,  # Background samples for SHAP
    'n_test_samples': 5,  # Number of test sequences to explain
    'feature_names': ['Feature_1', 'Feature_2', 'Feature_3', 'Target_Lagged']
}

# Output settings
OUTPUT_CONFIG = {
    'model_name': 'lstm_forecaster.h5',
    'scaler_name': 'scaler.pkl',
    'metrics_file': 'metrics.txt',
    'report_file': 'analysis_report.txt',
    'verbose': True
}

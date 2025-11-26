"""
Model Training and Evaluation Module
Handles data preprocessing, model training, and performance evaluation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import pickle
import os
from config import (
    MODEL_CONFIG, SPLIT_CONFIG, DATA_DIR, 
    MODELS_DIR, RESULTS_DIR, OUTPUT_CONFIG
)
from model import create_model


class TimeSeriesPreprocessor:
    """Handles data preprocessing and sequence creation"""
    
    def __init__(self, lookback, forecast_horizon):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        
    def create_sequences(self, data, target_col=-1):
        """
        Create sequences for time series forecasting
        
        Args:
            data: Array of shape (n_samples, n_features)
            target_col: Index of target column
            
        Returns:
            X: Input sequences (n_sequences, lookback, n_features)
            y: Target sequences (n_sequences, forecast_horizon)
        """
        X, y = [], []
        
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            # Input sequence
            X.append(data[i:i + self.lookback])
            
            # Target sequence (future values of target variable)
            y.append(data[i + self.lookback:i + self.lookback + self.forecast_horizon, target_col])
        
        return np.array(X), np.array(y)
    
    def fit_transform(self, data):
        """Fit scaler and transform data"""
        return self.scaler.fit_transform(data)
    
    def transform(self, data):
        """Transform data using fitted scaler"""
        return self.scaler.transform(data)
    
    def inverse_transform(self, data):
        """Inverse transform data"""
        return self.scaler.inverse_transform(data)
    
    def save_scaler(self, filepath):
        """Save scaler to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    @staticmethod
    def load_scaler(filepath):
        """Load scaler from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def load_and_split_data(filepath):
    """Load data and split into train/val/test sets"""
    print("\n" + "="*60)
    print("Loading and Splitting Data")
    print("="*60)
    
    # Load data
    df = pd.read_csv(filepath)
    data = df.values
    
    # Calculate split indices
    n = len(data)
    train_end = int(n * SPLIT_CONFIG['train_ratio'])
    val_end = int(n * (SPLIT_CONFIG['train_ratio'] + SPLIT_CONFIG['val_ratio']))
    
    # Split data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"✓ Total samples: {n}")
    print(f"✓ Training samples: {len(train_data)} ({SPLIT_CONFIG['train_ratio']*100:.0f}%)")
    print(f"✓ Validation samples: {len(val_data)} ({SPLIT_CONFIG['val_ratio']*100:.0f}%)")
    print(f"✓ Test samples: {len(test_data)} ({SPLIT_CONFIG['test_ratio']*100:.0f}%)")
    
    return train_data, val_data, test_data


def prepare_sequences(train_data, val_data, test_data):
    """Prepare sequences for training"""
    print("\n" + "="*60)
    print("Preparing Sequences")
    print("="*60)
    
    preprocessor = TimeSeriesPreprocessor(
        MODEL_CONFIG['lookback_window'],
        MODEL_CONFIG['forecast_horizon']
    )
    
    # Normalize data
    train_scaled = preprocessor.fit_transform(train_data)
    val_scaled = preprocessor.transform(val_data)
    test_scaled = preprocessor.transform(test_data)
    
    # Create sequences
    X_train, y_train = preprocessor.create_sequences(train_scaled)
    X_val, y_val = preprocessor.create_sequences(val_scaled)
    X_test, y_test = preprocessor.create_sequences(test_scaled)
    
    print(f"✓ Training sequences: {X_train.shape}")
    print(f"✓ Validation sequences: {X_val.shape}")
    print(f"✓ Test sequences: {X_test.shape}")
    print(f"✓ Target shape: {y_train.shape}")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, OUTPUT_CONFIG['scaler_name'])
    preprocessor.save_scaler(scaler_path)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessor


def train_model(model, train_data, val_data):
    """Train the model with callbacks"""
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=MODEL_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MODEL_CONFIG['epochs'],
        batch_size=MODEL_CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training complete!")
    print(f"✓ Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history


def evaluate_model(model, test_data, preprocessor):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("Evaluating Model Performance")
    print("="*60)
    
    X_test, y_test = test_data
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics for each forecast horizon
    metrics = {}
    for i in range(MODEL_CONFIG['forecast_horizon']):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        metrics[f'step_{i+1}'] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    # Overall metrics
    rmse_overall = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
    mae_overall = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    r2_overall = r2_score(y_test.flatten(), y_pred.flatten())
    
    metrics['overall'] = {
        'RMSE': rmse_overall,
        'MAE': mae_overall,
        'R2': r2_overall
    }
    
    print(f"\nOverall Performance Metrics:")
    print(f"  RMSE: {rmse_overall:.6f}")
    print(f"  MAE:  {mae_overall:.6f}")
    print(f"  R²:   {r2_overall:.6f}")
    
    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, OUTPUT_CONFIG['metrics_file'])
    with open(metrics_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Model Performance Metrics\n")
        f.write("="*60 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  RMSE: {rmse_overall:.6f}\n")
        f.write(f"  MAE:  {mae_overall:.6f}\n")
        f.write(f"  R²:   {r2_overall:.6f}\n\n")
        
        f.write("Per-Step Metrics:\n")
        for step, step_metrics in metrics.items():
            if step != 'overall':
                f.write(f"\n{step}:\n")
                for metric_name, value in step_metrics.items():
                    f.write(f"  {metric_name}: {value:.6f}\n")
    
    print(f"✓ Metrics saved to: {metrics_path}")
    
    return metrics, y_pred


def run_training_pipeline():
    """Execute complete training pipeline"""
    print("\n" + "="*70)
    print(" "*15 + "LSTM TRAINING PIPELINE")
    print("="*70)
    
    # Load and split data
    data_path = os.path.join(DATA_DIR, 'multivariate_timeseries.csv')
    train_data, val_data, test_data = load_and_split_data(data_path)
    
    # Prepare sequences
    train_seq, val_seq, test_seq, preprocessor = prepare_sequences(
        train_data, val_data, test_data
    )
    
    # Create model
    input_shape = (MODEL_CONFIG['lookback_window'], train_data.shape[1])
    model = create_model(input_shape)
    
    # Train model
    history = train_model(model, train_seq, val_seq)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, OUTPUT_CONFIG['model_name'])
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Evaluate model
    metrics, predictions = evaluate_model(model, test_seq, preprocessor)
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETE")
    print("="*70)
    
    return model, preprocessor, test_seq, metrics, predictions


if __name__ == "__main__":
    run_training_pipeline()

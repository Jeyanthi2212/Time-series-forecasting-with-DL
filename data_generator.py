"""
Multivariate Time Series Data Generator
Generates synthetic time series with controllable dependencies and patterns
"""

import numpy as np
import pandas as pd
from config import DATA_CONFIG, DATA_DIR
import os


class TimeSeriesGenerator:
    """
    Generates synthetic multivariate time series data with:
    - Trend components
    - Seasonal patterns
    - Lagged dependencies between features
    - Controlled noise
    """
    
    def __init__(self, config):
        self.n_samples = config['n_samples']
        self.n_features = config['n_features']
        self.noise_level = config['noise_level']
        self.trend_strength = config['trend_strength']
        self.seasonal_period = config['seasonal_period']
        self.seasonal_amplitude = config['seasonal_amplitude']
        np.random.seed(config['random_seed'])
        
    def generate_base_components(self):
        """Generate trend and seasonal components"""
        t = np.arange(self.n_samples)
        
        # Trend component
        trend = self.trend_strength * t
        
        # Seasonal component
        seasonal = self.seasonal_amplitude * np.sin(2 * np.pi * t / self.seasonal_period)
        
        return trend, seasonal
    
    def generate_features(self):
        """
        Generate multivariate time series with dependencies
        
        Feature relationships:
        - Feature 1: Base feature with trend + seasonality
        - Feature 2: Depends on Feature 1 with lag + own pattern
        - Feature 3: Depends on Features 1 & 2 + own pattern
        - Target: Nonlinear combination of all features with lag
        """
        trend, seasonal = self.generate_base_components()
        t = np.arange(self.n_samples)
        
        # Initialize features array
        features = np.zeros((self.n_samples, self.n_features + 1))  # +1 for target
        
        # Feature 1: Primary driver with trend and seasonality
        features[:, 0] = (
            trend + 
            seasonal + 
            0.5 * np.sin(2 * np.pi * t / (self.seasonal_period * 2)) +
            self.noise_level * np.random.randn(self.n_samples)
        )
        
        # Feature 2: Depends on lagged Feature 1
        lag_1 = 5
        for i in range(lag_1, self.n_samples):
            features[i, 1] = (
                0.6 * features[i - lag_1, 0] +
                0.3 * np.cos(2 * np.pi * t[i] / (self.seasonal_period * 1.5)) +
                0.1 * trend[i] +
                self.noise_level * np.random.randn()
            )
        
        # Feature 3: Depends on Features 1 and 2
        lag_2 = 3
        for i in range(max(lag_1, lag_2), self.n_samples):
            features[i, 2] = (
                0.4 * features[i - lag_2, 0] +
                0.3 * features[i - lag_2, 1] +
                0.2 * seasonal[i] +
                self.noise_level * np.random.randn()
            )
        
        # Target: Nonlinear combination with lags
        lag_target = 10
        for i in range(lag_target, self.n_samples):
            features[i, 3] = (
                0.5 * features[i - lag_target, 0] +
                0.3 * features[i - 5, 1] +
                0.2 * features[i - 3, 2] +
                0.1 * features[i - lag_target, 0] * features[i - 5, 1] +  # Interaction term
                0.05 * features[i - 3, 2] ** 2 +  # Nonlinear term
                self.noise_level * np.random.randn()
            )
        
        # Remove initial samples with zero padding
        features = features[lag_target:, :]
        
        return features
    
    def save_data(self, features):
        """Save generated data to CSV"""
        columns = [f'feature_{i+1}' for i in range(self.n_features)] + ['target']
        df = pd.DataFrame(features, columns=columns)
        
        filepath = os.path.join(DATA_DIR, 'multivariate_timeseries.csv')
        df.to_csv(filepath, index=False)
        
        print(f"Data saved to: {filepath}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nStatistics:")
        print(df.describe())
        
        return df


def generate_dataset():
    """Main function to generate dataset"""
    print("="*60)
    print("Generating Multivariate Time Series Dataset")
    print("="*60)
    
    generator = TimeSeriesGenerator(DATA_CONFIG)
    features = generator.generate_features()
    df = generator.save_data(features)
    
    print("\n✓ Dataset generation complete!")
    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Features: {df.shape[1] - 1}")
    print(f"✓ Target variable: target")
    
    return df


if __name__ == "__main__":
    generate_dataset()

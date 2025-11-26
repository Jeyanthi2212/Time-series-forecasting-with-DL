"""
Explainable AI (XAI) Analysis using SHAP
Provides interpretability for LSTM model predictions
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from config import (
    XAI_CONFIG, MODEL_CONFIG, MODELS_DIR, 
    RESULTS_DIR, OUTPUT_CONFIG
)


class LSTMExplainer:
    """SHAP-based explainer for LSTM models"""
    
    def __init__(self, model, background_data, feature_names):
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = None
        
    def create_explainer(self):
        """Create SHAP explainer"""
        print("\n" + "="*60)
        print("Creating SHAP Explainer")
        print("="*60)
        
        # Use DeepExplainer for neural networks
        self.explainer = shap.DeepExplainer(
            self.model,
            self.background_data
        )
        
        print(f"✓ Explainer created with {len(self.background_data)} background samples")
        
    def explain_predictions(self, test_sequences, n_samples=5):
        """
        Generate SHAP values for test sequences
        
        Args:
            test_sequences: Test data to explain
            n_samples: Number of samples to explain
            
        Returns:
            shap_values: SHAP values for each prediction
        """
        print("\n" + "="*60)
        print("Computing SHAP Values")
        print("="*60)
        
        if self.explainer is None:
            self.create_explainer()
        
        # Select samples to explain
        indices = np.random.choice(len(test_sequences), n_samples, replace=False)
        samples = test_sequences[indices]
        
        print(f"✓ Analyzing {n_samples} test samples")
        print(f"✓ Input shape: {samples.shape}")
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(samples)
        
        # Handle multiple outputs (forecast horizons)
        if isinstance(shap_values, list):
            print(f"✓ SHAP values computed for {len(shap_values)} output steps")
        else:
            print(f"✓ SHAP values computed: {shap_values.shape}")
        
        return shap_values, samples, indices
    
    def analyze_feature_importance(self, shap_values, samples):
        """
        Analyze overall feature importance across timesteps
        
        Returns:
            Dictionary with feature importance metrics
        """
        print("\n" + "="*60)
        print("Analyzing Feature Importance")
        print("="*60)
        
        # Handle multiple outputs (take first forecast step)
        if isinstance(shap_values, list):
            shap_array = np.array(shap_values[0])
        else:
            shap_array = shap_values
        
        # Compute mean absolute SHAP value for each feature across all timesteps
        feature_importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Mean absolute SHAP across samples and timesteps
            importance = np.mean(np.abs(shap_array[:, :, i]))
            feature_importance[feature_name] = importance
        
        # Normalize to percentages
        total = sum(feature_importance.values())
        feature_importance_pct = {
            k: (v / total) * 100 
            for k, v in feature_importance.items()
        }
        
        print("\nFeature Importance (% contribution):")
        for feature, importance in sorted(
            feature_importance_pct.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            print(f"  {feature:20s}: {importance:6.2f}%")
        
        return feature_importance_pct
    
    def analyze_temporal_patterns(self, shap_values, samples):
        """Analyze importance across timesteps"""
        print("\n" + "="*60)
        print("Analyzing Temporal Patterns")
        print("="*60)
        
        if isinstance(shap_values, list):
            shap_array = np.array(shap_values[0])
        else:
            shap_array = shap_values
        
        # Compute importance for each timestep
        timestep_importance = np.mean(np.abs(shap_array), axis=(0, 2))
        
        print("\nTemporal Importance Pattern:")
        lookback = MODEL_CONFIG['lookback_window']
        recent_steps = min(10, lookback)
        
        print(f"  Most recent {recent_steps} timesteps:")
        for i in range(recent_steps):
            idx = lookback - i - 1
            print(f"    t-{i}: {timestep_importance[idx]:.6f}")
        
        return timestep_importance
    
    def generate_insights(self, shap_values, samples, feature_importance, temporal_importance):
        """Generate key insights from XAI analysis"""
        print("\n" + "="*60)
        print("Generating Insights")
        print("="*60)
        
        insights = []
        
        # Insight 1: Most important feature
        top_feature = max(feature_importance.items(), key=lambda x: x[1])
        insights.append({
            'title': 'Primary Predictive Feature',
            'description': f"{top_feature[0]} is the most influential feature, contributing "
                          f"{top_feature[1]:.1f}% to the model's predictions. This suggests "
                          f"that variations in {top_feature[0]} are the strongest driver of "
                          f"future target values."
        })
        
        # Insight 2: Temporal decay pattern
        recent_importance = np.mean(temporal_importance[-5:])
        older_importance = np.mean(temporal_importance[:5])
        decay_ratio = recent_importance / (older_importance + 1e-8)
        
        if decay_ratio > 1.5:
            insights.append({
                'title': 'Recent Data Emphasis',
                'description': f"The model places {decay_ratio:.1f}x more importance on recent "
                              f"observations (last 5 timesteps) compared to older data. This "
                              f"indicates the forecast is primarily driven by recent trends "
                              f"rather than long-term patterns."
            })
        else:
            insights.append({
                'title': 'Balanced Temporal Attention',
                'description': f"The model distributes attention relatively evenly across the "
                              f"lookback window (ratio: {decay_ratio:.2f}), suggesting both "
                              f"recent and historical data contribute meaningfully to predictions."
            })
        
        # Insight 3: Feature interactions
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_2_contribution = sum([f[1] for f in sorted_features[:2]])
        
        if top_2_contribution > 70:
            insights.append({
                'title': 'Dominant Feature Pair',
                'description': f"The top two features ({sorted_features[0][0]} and "
                              f"{sorted_features[1][0]}) account for {top_2_contribution:.1f}% "
                              f"of predictive power. This suggests these features capture the "
                              f"primary dynamics of the target variable."
            })
        else:
            insights.append({
                'title': 'Distributed Feature Importance',
                'description': f"Predictive power is distributed across multiple features "
                              f"(top 2 contribute {top_2_contribution:.1f}%). This indicates "
                              f"complex interactions between features are important for "
                              f"accurate forecasting."
            })
        
        # Insight 4: Lagged target importance
        if 'Target_Lagged' in feature_importance:
            lag_importance = feature_importance['Target_Lagged']
            if lag_importance

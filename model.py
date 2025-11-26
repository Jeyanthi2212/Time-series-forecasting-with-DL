"""
LSTM Model Architecture with Attention Mechanism
Implements a production-quality sequence model for time series forecasting
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import MODEL_CONFIG


class AttentionLayer(layers.Layer):
    """Custom attention layer for sequence models"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Compute attention scores
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        # Apply attention weights
        output = inputs * a
        return tf.reduce_sum(output, axis=1)


def build_lstm_model(input_shape, config):
    """
    Build LSTM model with optional attention mechanism
    
    Args:
        input_shape: Tuple (timesteps, features)
        config: Model configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential(name='LSTM_Forecaster')
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # LSTM layers
    for i, units in enumerate(config['lstm_units']):
        return_sequences = (i < len(config['lstm_units']) - 1) or config['attention']
        
        model.add(layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=config['dropout_rate'],
            recurrent_dropout=config['dropout_rate'] * 0.5,
            name=f'lstm_{i+1}'
        ))
        
        # Batch normalization for training stability
        if return_sequences:
            model.add(layers.BatchNormalization(name=f'bn_lstm_{i+1}'))
    
    # Attention mechanism
    if config['attention']:
        model.add(AttentionLayer(name='attention'))
        model.add(layers.BatchNormalization(name='bn_attention'))
    
    # Dense layers for output
    model.add(layers.Dense(64, activation='relu', name='dense_1'))
    model.add(layers.Dropout(config['dropout_rate'], name='dropout_dense'))
    model.add(layers.Dense(32, activation='relu', name='dense_2'))
    
    # Output layer (forecasting horizon)
    model.add(layers.Dense(config['forecast_horizon'], name='output'))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def create_model(input_shape):
    """Create and return the model"""
    print("\n" + "="*60)
    print("Building LSTM Model Architecture")
    print("="*60)
    
    model = build_lstm_model(input_shape, MODEL_CONFIG)
    
    print(f"\n✓ Model created successfully")
    print(f"✓ Input shape: {input_shape}")
    print(f"✓ Forecast horizon: {MODEL_CONFIG['forecast_horizon']} steps")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    print("\nModel Architecture:")
    model.summary()
    
    return model


if __name__ == "__main__":
    # Test model creation
    test_shape = (MODEL_CONFIG['lookback_window'], 4)  # 4 features
    model = create_model(test_shape)

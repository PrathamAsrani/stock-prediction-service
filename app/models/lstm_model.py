import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from app.models.base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class LSTMModel(BaseModel):
    """LSTM model for time series pattern recognition"""
    
    def __init__(self, sequence_length: int = 60):
        super().__init__(model_name="lstm")
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.build_model()
    
    def build_model(self):
        """Build LSTM architecture"""
        # Will be built dynamically based on input shape
        self.model = None
        logger.info("LSTM model will be built on first training")
    
    def _build_architecture(self, input_shape: Tuple):
        """Build LSTM architecture with given input shape"""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                128, 
                return_sequences=True, 
                input_shape=input_shape
            ),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # Third LSTM layer
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        logger.info("LSTM architecture built")
        return model
    
    def _create_sequences(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for LSTM input"""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X) if not self.is_trained else self.scaler.transform(X)
        
        sequences = []
        targets = [] if y is not None else None
        
        for i in range(len(X_scaled) - self.sequence_length):
            sequences.append(X_scaled[i:i + self.sequence_length])
            if y is not None:
                targets.append(y.iloc[i + self.sequence_length])
        
        X_seq = np.array(sequences)
        y_seq = np.array(targets) if targets is not None else None
        
        return X_seq, y_seq
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict:
        """Train LSTM model"""
        logger.info(f"Training LSTM model with {len(X_train)} samples")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        
        logger.info(f"Created {len(X_train_seq)} sequences of length {self.sequence_length}")
        
        # Build model architecture
        if self.model is None:
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            self.model = self._build_architecture(input_shape)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = self.model.fit(
            X_train_seq, 
            y_train_seq,
            epochs=100,
            batch_size=32,
            validation_data=validation_data,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        
        # Evaluate
        train_metrics = self._evaluate_sequences(X_train_seq, y_train_seq)
        self.metrics = train_metrics
        
        if validation_data is not None:
            val_metrics = self._evaluate_sequences(X_val_seq, y_val_seq)
            self.metrics['validation'] = val_metrics
        
        logger.info(f"LSTM training complete. Accuracy: {train_metrics['accuracy']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure features match
        X = X[self.feature_names]
        
        # Create sequences
        X_seq, _ = self._create_sequences(X, None)
        
        # Predict
        probabilities = self.model.predict(X_seq).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def _evaluate_sequences(self, X_seq: np.ndarray, y_seq: np.ndarray) -> Dict:
        """Evaluate model on sequences"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        probabilities = self.model.predict(X_seq).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_seq, predictions),
            'precision': precision_score(y_seq, predictions, zero_division=0),
            'recall': recall_score(y_seq, predictions, zero_division=0),
            'f1_score': f1_score(y_seq, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y_seq, probabilities) if len(np.unique(y_seq)) > 1 else 0.0
        }
        
        return metrics
    
    def save(self, filename: Optional[str] = None):
        """Save LSTM model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filename is None:
            filename = "lstm_model.h5"
        
        model_file = self.model_path / filename
        scaler_file = self.model_path / "lstm_scaler.pkl"
        
        # Save model
        self.model.save(model_file)
        
        # Save scaler and metadata
        import joblib
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'metrics': self.metrics
        }
        joblib.dump(metadata, scaler_file)
        
        logger.info(f"LSTM model saved to {model_file}")
    
    def load(self, filename: Optional[str] = None):
        """Load LSTM model"""
        if filename is None:
            filename = "lstm_model.h5"
        
        model_file = self.model_path / filename
        scaler_file = self.model_path / "lstm_scaler.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load model
        self.model = keras.models.load_model(model_file)
        
        # Load metadata
        import joblib
        metadata = joblib.load(scaler_file)
        self.scaler = metadata['scaler']
        self.feature_names = metadata['feature_names']
        self.sequence_length = metadata['sequence_length']
        self.metrics = metadata['metrics']
        self.is_trained = True
        
        logger.info(f"LSTM model loaded from {model_file}")
        
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer, TFAutoModel
from app.models.base_model import BaseModel
from app.services.rag_service import RAGService
import logging

logger = logging.getLogger(__name__)

class TransformerModel(BaseModel):
    """Transformer model for financial report and news analysis"""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        max_length: int = 512
    ):
        super().__init__(model_name="transformer")
        self.transformer_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.rag_service = RAGService()
        self.build_model()
    
    def build_model(self):
        """Build Transformer architecture"""
        logger.info(f"Loading tokenizer: {self.transformer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_name)
        # Model will be built dynamically
        self.model = None
    
    def _build_architecture(self, num_features: int):
        """Build Transformer-based architecture"""
        
        # Text input
        input_ids = layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = layers.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        # Numerical features input
        numerical_input = layers.Input(shape=(num_features,), name='numerical_features')
        
        # Load pre-trained transformer
        transformer = TFAutoModel.from_pretrained(self.transformer_name)
        
        # Get transformer outputs
        transformer_output = transformer(input_ids, attention_mask=attention_mask)
        
        # Use CLS token representation
        text_features = transformer_output.last_hidden_state[:, 0, :]
        
        # Process text features
        text_branch = layers.Dense(256, activation='relu')(text_features)
        text_branch = layers.Dropout(0.3)(text_branch)
        text_branch = layers.Dense(128, activation='relu')(text_branch)
        text_branch = layers.Dropout(0.3)(text_branch)
        
        # Process numerical features
        numerical_branch = layers.Dense(128, activation='relu')(numerical_input)
        numerical_branch = layers.Dropout(0.2)(numerical_branch)
        numerical_branch = layers.Dense(64, activation='relu')(numerical_branch)
        numerical_branch = layers.Dropout(0.2)(numerical_branch)
        
        # Concatenate both branches
        combined = layers.Concatenate()([text_branch, numerical_branch])
        
        # Final layers
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = keras.Model(
            inputs=[input_ids, attention_mask, numerical_input],
            outputs=output
        )
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        logger.info("Transformer architecture built")
        return model
    
    def _prepare_text_data(
        self, 
        symbols: List[str], 
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare text data from RAG service"""
        
        input_ids_list = []
        attention_mask_list = []
        
        for symbol in symbols:
            # Get relevant context from knowledge base
            context = self.rag_service.vector_store.get_relevant_context(
                query=f"swing trading analysis {symbol}",
                symbol=symbol,
                max_tokens=400  # Shorter for transformer
            )
            
            # Tokenize
            encoded = self.tokenizer(
                context,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            
            input_ids_list.append(encoded['input_ids'][0])
            attention_mask_list.append(encoded['attention_mask'][0])
        
        input_ids = np.array(input_ids_list)
        attention_mask = np.array(attention_mask_list)
        
        return input_ids, attention_mask
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        symbols_train: List[str],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        symbols_val: Optional[List[str]] = None
    ) -> Dict:
        """Train Transformer model"""
        logger.info(f"Training Transformer model with {len(X_train)} samples")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Prepare text data
        logger.info("Preparing text data from knowledge base...")
        input_ids_train, attention_mask_train = self._prepare_text_data(symbols_train, X_train)
        
        # Build model if not exists
        if self.model is None:
            self.model = self._build_architecture(num_features=X_train.shape[1])
        
        # Prepare training data
        train_data = {
            'input_ids': input_ids_train,
            'attention_mask': attention_mask_train,
            'numerical_features': X_train.values
        }
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None and symbols_val is not None:
            input_ids_val, attention_mask_val = self._prepare_text_data(symbols_val, X_val)
            validation_data = (
                {
                    'input_ids': input_ids_val,
                    'attention_mask': attention_mask_val,
                    'numerical_features': X_val.values
                },
                y_val.values
            )
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # Train
        history = self.model.fit(
            train_data,
            y_train.values,
            epochs=20,  # Fewer epochs for transformer
            batch_size=16,  # Smaller batch size
            validation_data=validation_data,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        
        # Evaluate
        logger.info("Evaluating transformer model...")
        # Note: This is simplified - full evaluation would need symbols
        self.metrics = {'accuracy': history.history['accuracy'][-1]}
        
        logger.info(f"Transformer training complete.")
        
        return self.metrics
    
    def predict(
        self, 
        X: pd.DataFrame, 
        symbols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure features match
        X = X[self.feature_names]
        
        # Prepare text data
        input_ids, attention_mask = self._prepare_text_data(symbols, X)
        
        # Prepare input
        test_data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'numerical_features': X.values
        }
        
        # Predict
        probabilities = self.model.predict(test_data).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def save(self, filename: Optional[str] = None):
        """Save Transformer model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filename is None:
            filename = "transformer_model.h5"
        
        model_file = self.model_path / filename
        
        # Save model
        self.model.save(model_file)
        
        # Save metadata
        import joblib
        metadata = {
            'feature_names': self.feature_names,
            'transformer_name': self.transformer_name,
            'max_length': self.max_length,
            'metrics': self.metrics
        }
        joblib.dump(metadata, self.model_path / "transformer_metadata.pkl")
        
        logger.info(f"Transformer model saved to {model_file}")
    
    def load(self, filename: Optional[str] = None):
        """Load Transformer model"""
        if filename is None:
            filename = "transformer_model.h5"
        
        model_file = self.model_path / filename
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load model
        self.model = keras.models.load_model(model_file)
        
        # Load metadata
        import joblib
        metadata = joblib.load(self.model_path / "transformer_metadata.pkl")
        self.feature_names = metadata['feature_names']
        self.transformer_name = metadata['transformer_name']
        self.max_length = metadata['max_length']
        self.metrics = metadata['metrics']
        self.is_trained = True
        
        logger.info(f"Transformer model loaded from {model_file}")
        
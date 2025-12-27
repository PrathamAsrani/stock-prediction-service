from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.model_path = Path(f"trained_models/{model_name}")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.feature_names = []
        self.metrics = {}
    
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        Returns: (predictions, probabilities)
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        predictions, probabilities = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1_score': f1_score(y, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        logger.info(f"{self.model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save(self, filename: Optional[str] = None):
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filename is None:
            filename = f"{self.model_name}_model.pkl"
        
        model_file = self.model_path / filename
        
        save_dict = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(save_dict, model_file)
        logger.info(f"Model saved to {model_file}")
    
    def load(self, filename: Optional[str] = None):
        """Load model from disk"""
        if filename is None:
            filename = f"{self.model_name}_model.pkl"
        
        model_file = self.model_path / filename
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        save_dict = joblib.load(model_file)
        
        self.model = save_dict['model']
        self.feature_names = save_dict['feature_names']
        self.metrics = save_dict['metrics']
        self.is_trained = save_dict['is_trained']
        
        logger.info(f"Model loaded from {model_file}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (if supported by model)"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        logger.warning(f"{self.model_name} does not support feature importance")
        return pd.DataFrame()
    
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from app.models.base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost model for technical indicator-based predictions"""
    
    def __init__(self):
        super().__init__(model_name="xgboost")
        self.scaler = StandardScaler()
        self.build_model()
    
    def build_model(self):
        """Build XGBoost classifier"""
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=20,
            tree_method='hist',
            enable_categorical=False
        )
        logger.info("XGBoost model architecture created")
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict:
        """Train XGBoost model"""
        logger.info(f"Training XGBoost model with {len(X_train)} samples")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation set if provided
        eval_set = []
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        # Train
        self.model.fit(
            X_train_scaled, 
            y_train,
            eval_set=eval_set if eval_set else None,
            verbose=True
        )
        
        self.is_trained = True
        
        # Evaluate
        self.metrics = self.evaluate(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.metrics['validation'] = val_metrics
        
        logger.info(f"XGBoost training complete. F1 Score: {self.metrics['f1_score']:.4f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure features match
        X = X[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance with XGBoost-specific metrics"""
        if not self.is_trained:
            return pd.DataFrame()
        
        # Get importance scores
        importance_gain = self.model.get_booster().get_score(importance_type='gain')
        importance_weight = self.model.get_booster().get_score(importance_type='weight')
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_gain': [importance_gain.get(f'f{i}', 0) for i in range(len(self.feature_names))],
            'importance_weight': [importance_weight.get(f'f{i}', 0) for i in range(len(self.feature_names))]
        })
        
        # Sort by gain
        importance_df = importance_df.sort_values('importance_gain', ascending=False)
        
        return importance_df
    
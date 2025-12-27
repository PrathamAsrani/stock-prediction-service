import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class StockPredictionModel:
    """ML Model for stock price prediction"""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.model_path = Path("trained_models")
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize model based on type
        if model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.05,
                reg_lambda=1,
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        
        # Select feature columns (exclude target and metadata)
        feature_cols = [
            'returns', 'log_returns', 'volatility', 'momentum',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'sma_20', 'sma_50', 'ema_12',
            'bb_upper', 'bb_lower', 'bb_width',
            'volume_sma', 'volume_ratio',
            'atr', 'stoch_k', 'stoch_d', 'adx', 'obv',
            'price_to_bb', 'trend_strength'
        ]
        
        # Add lag features
        for lag in [1, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            feature_cols.extend([f'close_lag_{lag}', f'volume_lag_{lag}', f'returns_lag_{lag}'])
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'returns_mean_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
            feature_cols.extend([
                f'close_std_{window}',
                f'returns_mean_{window}',
                f'returns_std_{window}'
            ])
        
        # Add fundamental features if available
        if 'pe_ratio' in df.columns:
            fundamental_cols = ['pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 'dividend_yield']
            feature_cols.extend([col for col in fundamental_cols if col in df.columns])
        
        self.feature_columns = feature_cols
        
        return df[feature_cols]
    
    def create_target(self, df: pd.DataFrame, days_ahead: int = 30, target_return: float = 0.10) -> pd.Series:
        """
        Create binary target: 1 if stock grows >= target_return% in days_ahead days, 0 otherwise
        """
        # Calculate future return
        future_price = df['close'].shift(-days_ahead)
        future_return = (future_price - df['close']) / df['close']
        
        # Binary target: 1 if return >= target_return, 0 otherwise
        target = (future_return >= target_return).astype(int)
        
        return target
    
    def train(
        self,
        df: pd.DataFrame,
        days_ahead: int = 30,
        target_return: float = 0.10,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """Train the model"""
        
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = self.create_target(df, days_ahead, target_return)
        
        # Remove rows with NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Training samples: {len(X)}, Positive samples: {y.sum()}, Negative samples: {(~y.astype(bool)).sum()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_rate': float(y.mean())
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='f1')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        logger.info(f"Model trained successfully. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        self.is_trained = True
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Make prediction
        
        Returns:
            Tuple of (prediction, confidence)
            prediction: 1 if will grow, 0 otherwise
            confidence: probability score
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        # Ensure features match training features
        features = features[self.feature_columns]
        
        # Handle NaN values
        features = features.fillna(features.mean())
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0][1]
        
        return int(prediction), float(confidence)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return pd.DataFrame()
    
    def save_model(self, filename: Optional[str] = None):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")
        
        if filename is None:
            filename = f"{self.model_type}_model.pkl"
        
        model_file = self.model_path / filename
        scaler_file = self.model_path / f"{self.model_type}_scaler.pkl"
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.feature_columns, self.model_path / f"{self.model_type}_features.pkl")
        
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self, filename: Optional[str] = None):
        """Load trained model from disk"""
        if filename is None:
            filename = f"{self.model_type}_model.pkl"
        
        model_file = self.model_path / filename
        scaler_file = self.model_path / f"{self.model_type}_scaler.pkl"
        features_file = self.model_path / f"{self.model_type}_features.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.feature_columns = joblib.load(features_file)
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_file}")
        
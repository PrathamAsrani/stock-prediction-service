import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from app.models.base_model import BaseModel
from app.models.xgboost_model import XGBoostModel
from app.models.lstm_model import LSTMModel
from app.models.transformer_model import TransformerModel
import logging

logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """Ensemble model combining XGBoost, LSTM, and Transformer"""
    
    def __init__(
        self,
        use_xgboost: bool = True,
        use_lstm: bool = True,
        use_transformer: bool = True,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(model_name="ensemble")
        
        self.use_xgboost = use_xgboost
        self.use_lstm = use_lstm
        self.use_transformer = use_transformer
        
        # Default weights (can be tuned)
        self.weights = weights or {
            'xgboost': 0.4,
            'lstm': 0.35,
            'transformer': 0.25
        }
        
        # Initialize base models
        self.base_models = {}
        if use_xgboost:
            self.base_models['xgboost'] = XGBoostModel()
        if use_lstm:
            self.base_models['lstm'] = LSTMModel()
        if use_transformer:
            self.base_models['transformer'] = TransformerModel()
        
        # Meta-learner
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
        logger.info(f"Ensemble initialized with models: {list(self.base_models.keys())}")
    
    def build_model(self):
        """Build ensemble - models are already initialized"""
        pass
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        symbols_train: Optional[List[str]] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        symbols_val: Optional[List[str]] = None
    ) -> Dict:
        """Train ensemble model"""
        logger.info("="*60)
        logger.info("STARTING ENSEMBLE TRAINING")
        logger.info("="*60)
        
        self.feature_names = list(X_train.columns)
        
        # Store base model predictions
        train_meta_features = []
        val_meta_features = [] if X_val is not None else None
        
        ensemble_metrics = {
            'base_models': {},
            'ensemble': {}
        }
        
        # Train each base model
        for model_name, model in self.base_models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name.upper()} model")
            logger.info(f"{'='*60}")
            
            try:
                if model_name == 'transformer' and symbols_train is not None:
                    # Transformer needs symbols for RAG
                    metrics = model.train(
                        X_train, y_train, 
                        symbols_train=symbols_train,
                        X_val=X_val, y_val=y_val,
                        symbols_val=symbols_val
                    )
                else:
                    metrics = model.train(X_train, y_train, X_val, y_val)
                
                ensemble_metrics['base_models'][model_name] = metrics
                
                # Get predictions for meta-learner
                if model_name == 'transformer' and symbols_train is not None:
                    _, train_probs = model.predict(X_train, symbols_train)
                    train_meta_features.append(train_probs)
                    
                    if X_val is not None and symbols_val is not None:
                        _, val_probs = model.predict(X_val, symbols_val)
                        val_meta_features.append(val_probs)
                else:
                    _, train_probs = model.predict(X_train)
                    train_meta_features.append(train_probs)
                    
                    if X_val is not None:
                        _, val_probs = model.predict(X_val)
                        val_meta_features.append(val_probs)
                
                logger.info(f"{model_name} training successful!")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                # Continue with other models
                continue
        
        # Prepare meta-features
        if len(train_meta_features) > 0:
            X_meta_train = np.column_stack(train_meta_features)
            
            if val_meta_features and len(val_meta_features) > 0:
                X_meta_val = np.column_stack(val_meta_features)
            else:
                X_meta_val = None
            
            # Train meta-learner
            logger.info(f"\n{'='*60}")
            logger.info("Training Meta-Learner (Stacking)")
            logger.info(f"{'='*60}")
            
            X_meta_train_scaled = self.scaler.fit_transform(X_meta_train)
            self.meta_learner.fit(X_meta_train_scaled, y_train)
            
            # Evaluate ensemble
            ensemble_train_pred = self.meta_learner.predict(X_meta_train_scaled)
            ensemble_train_proba = self.meta_learner.predict_proba(X_meta_train_scaled)[:, 1]
            
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            
            ensemble_metrics['ensemble']['train'] = {
                'accuracy': accuracy_score(y_train, ensemble_train_pred),
                'f1_score': f1_score(y_train, ensemble_train_pred),
                'roc_auc': roc_auc_score(y_train, ensemble_train_proba)
            }
            
            if X_meta_val is not None and y_val is not None:
                X_meta_val_scaled = self.scaler.transform(X_meta_val)
                ensemble_val_pred = self.meta_learner.predict(X_meta_val_scaled)
                ensemble_val_proba = self.meta_learner.predict_proba(X_meta_val_scaled)[:, 1]
                
                ensemble_metrics['ensemble']['validation'] = {
                    'accuracy': accuracy_score(y_val, ensemble_val_pred),
                    'f1_score': f1_score(y_val, ensemble_val_pred),
                    'roc_auc': roc_auc_score(y_val, ensemble_val_proba)
                }
            
            self.is_trained = True
            self.metrics = ensemble_metrics
            
            logger.info(f"\n{'='*60}")
            logger.info("ENSEMBLE TRAINING COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"Train Accuracy: {ensemble_metrics['ensemble']['train']['accuracy']:.4f}")
            logger.info(f"Train F1 Score: {ensemble_metrics['ensemble']['train']['f1_score']:.4f}")
            
            if 'validation' in ensemble_metrics['ensemble']:
                logger.info(f"Val Accuracy: {ensemble_metrics['ensemble']['validation']['accuracy']:.4f}")
                logger.info(f"Val F1 Score: {ensemble_metrics['ensemble']['validation']['f1_score']:.4f}")
            
            return ensemble_metrics
        else:
            raise ValueError("No base models were successfully trained")
    
    def predict(
        self, 
        X: pd.DataFrame,
        symbols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble model not trained yet")
        
        # Get predictions from each base model
        meta_features = []
        
        for model_name, model in self.base_models.items():
            if not model.is_trained:
                logger.warning(f"{model_name} not trained, skipping...")
                continue
            
            try:
                if model_name == 'transformer' and symbols is not None:
                    _, probs = model.predict(X, symbols)
                else:
                    _, probs = model.predict(X)
                
                meta_features.append(probs)
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {str(e)}")
                continue
        
        if len(meta_features) == 0:
            raise ValueError("No base models available for prediction")
        
        # Stack predictions
        X_meta = np.column_stack(meta_features)
        X_meta_scaled = self.scaler.transform(X_meta)
        
        # Meta-learner prediction
        probabilities = self.meta_learner.predict_proba(X_meta_scaled)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def predict_with_breakdown(
        self,
        X: pd.DataFrame,
        symbols: Optional[List[str]] = None
    ) -> Dict:
        """Get predictions with breakdown from each model"""
        
        breakdown = {
            'base_predictions': {},
            'ensemble_prediction': None,
            'ensemble_probability': None
        }
        
        meta_features = []
        
        for model_name, model in self.base_models.items():
            if not model.is_trained:
                continue
            
            try:
                if model_name == 'transformer' and symbols is not None:
                    preds, probs = model.predict(X, symbols)
                else:
                    preds, probs = model.predict(X)
                
                breakdown['base_predictions'][model_name] = {
                    'prediction': preds[0] if len(preds) > 0 else None,
                    'probability': probs[0] if len(probs) > 0 else None
                }
                
                meta_features.append(probs)
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {str(e)}")
                continue
        
        # Ensemble prediction
        if len(meta_features) > 0:
            X_meta = np.column_stack(meta_features)
            X_meta_scaled = self.scaler.transform(X_meta)
            
            ensemble_prob = self.meta_learner.predict_proba(X_meta_scaled)[:, 1][0]
            ensemble_pred = int(ensemble_prob > 0.5)
            
            breakdown['ensemble_prediction'] = ensemble_pred
            breakdown['ensemble_probability'] = ensemble_prob
        
        return breakdown
    
    def save(self, filename: Optional[str] = None):
        """Save ensemble model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Save each base model
        for model_name, model in self.base_models.items():
            if model.is_trained:
                model.save()
        
        # Save meta-learner and ensemble metadata
        import joblib
        
        ensemble_data = {
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'weights': self.weights,
            'use_xgboost': self.use_xgboost,
            'use_lstm': self.use_lstm,
            'use_transformer': self.use_transformer,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        joblib.dump(ensemble_data, self.model_path / "ensemble_meta.pkl")
        
        logger.info("Ensemble model saved successfully")
    
    def load(self, filename: Optional[str] = None):
        """Load ensemble model"""
        import joblib
        
        # Load each base model
        for model_name, model in self.base_models.items():
            try:
                model.load()
                logger.info(f"Loaded {model_name} model")
            except FileNotFoundError:
                logger.warning(f"{model_name} model not found")
        
        # Load meta-learner
        meta_file = self.model_path / "ensemble_meta.pkl"
        if meta_file.exists():
            ensemble_data = joblib.load(meta_file)
            
            self.meta_learner = ensemble_data['meta_learner']
            self.scaler = ensemble_data['scaler']
            self.weights = ensemble_data['weights']
            self.use_xgboost = ensemble_data['use_xgboost']
            self.use_lstm = ensemble_data['use_lstm']
            self.use_transformer = ensemble_data['use_transformer']
            self.feature_names = ensemble_data['feature_names']
            self.metrics = ensemble_data['metrics']
            self.is_trained = True
            
            logger.info("Ensemble model loaded successfully")
        else:
            raise FileNotFoundError("Ensemble metadata not found")
        
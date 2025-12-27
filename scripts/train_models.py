import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from app.models.ensemble_model import EnsembleModel
from app.features.technical_features import TechnicalAnalyzer
from app.services.data_service import DataService
from app.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Complete model training pipeline"""
    
    def __init__(self):
        self.data_service = DataService()
        self.technical_analyzer = TechnicalAnalyzer()
        
        self.ensemble_model = EnsembleModel(
            use_xgboost=settings.ENABLE_XGBOOST,
            use_lstm=settings.ENABLE_LSTM,
            use_transformer=settings.ENABLE_TRANSFORMER
        )
    
    async def collect_training_data(self, symbols: list) -> pd.DataFrame:
        """Collect and prepare training data"""
        logger.info("="*60)
        logger.info("COLLECTING TRAINING DATA")
        logger.info("="*60)
        
        all_data = []
        symbols_list = []
        
        for symbol in tqdm(symbols, desc="Fetching data"):
            try:
                # Fetch historical data
                candles = self.data_service.fetch_historical_data(
                    symbol=symbol,
                    exchange="NSE",
                    days=365 * settings.YEARS_OF_HISTORY
                )
                
                if not candles or len(candles) < 100:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = self.data_service.candles_to_dataframe(candles)
                
                # Calculate technical features
                df = TechnicalAnalyzer.calculate_features(df)
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Create target (10% growth in 30 days)
                future_price = df['close'].shift(-30)
                df['target'] = ((future_price - df['close']) / df['close'] >= 0.10).astype(int)
                
                # Remove rows with NaN
                df = df.dropna()
                
                if len(df) > 0:
                    all_data.append(df)
                    symbols_list.extend([symbol] * len(df))
                    logger.info(f"✓ {symbol}: {len(df)} samples")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No training data collected")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"\n✓ Total samples collected: {len(combined_df)}")
        logger.info(f"✓ Positive samples: {combined_df['target'].sum()}")
        logger.info(f"✓ Negative samples: {len(combined_df) - combined_df['target'].sum()}")
        logger.info(f"✓ Features: {len(combined_df.columns) - 2}")  # -2 for symbol and target
        
        return combined_df, symbols_list
    
    async def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target"""
        
        # Define feature columns (exclude metadata and target)
        exclude_cols = ['symbol', 'target', 'timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['target']
        symbols = df['symbol'].values
        
        logger.info(f"Features prepared: {len(feature_cols)} features")
        
        return X, y, symbols
    
    async def train_ensemble(self, symbols_list: list):
        """Train complete ensemble model"""
        logger.info("\n" + "="*60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*60 + "\n")
        
        # Collect data
        df, symbols_data = await self.collect_training_data(symbols_list)
        
        # Prepare features
        X, y, symbols = await self.prepare_features(df)
        
        # Split data
        logger.info("\nSplitting data...")
        (
            X_train, X_temp, 
            y_train, y_temp,
            symbols_train, symbols_temp
        ) = train_test_split(
            X, y, symbols,
            test_size=settings.TRAIN_TEST_SPLIT + settings.VALIDATION_SPLIT,
            random_state=settings.RANDOM_SEED,
            stratify=y
        )
        
        (
            X_val, X_test,
            y_val, y_test,
            symbols_val, symbols_test
        ) = train_test_split(
            X_temp, y_temp, symbols_temp,
            test_size=0.5,
            random_state=settings.RANDOM_SEED,
            stratify=y_temp
        )
        
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # Train ensemble
        metrics = self.ensemble_model.train(
            X_train=X_train,
            y_train=y_train,
            symbols_train=symbols_train.tolist(),
            X_val=X_val,
            y_val=y_val,
            symbols_val=symbols_val.tolist()
        )
        
        # Test evaluation
        logger.info("\n" + "="*60)
        logger.info("FINAL TEST EVALUATION")
        logger.info("="*60)
        
        test_preds, test_probs = self.ensemble_model.predict(
            X_test, 
            symbols=symbols_test.tolist()
        )
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nClassification Report:")
        print(classification_report(y_test, test_preds))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, test_preds))
        
        # Save model
        logger.info("\nSaving ensemble model...")
        self.ensemble_model.save()
        
        logger.info("\n" + "="*60)
        logger.info("✓ MODEL TRAINING COMPLETE!")
        logger.info("="*60)
        
        return metrics

async def main():
    """Main training function"""
    
    # Training symbols (expand as needed)
    training_symbols = [
        # Large Cap
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
        'ICICIBANK', 'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC',
        'BAJFINANCE', 'LT', 'ASIANPAINT', 'AXISBANK', 'MARUTI',
        
        # Mid Cap
        'ADANIPORTS', 'TATAMOTORS', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO',
        'NESTLEIND', 'POWERGRID', 'NTPC', 'ONGC', 'M&M',
        
        # Additional diversity
        'WIPRO', 'TECHM', 'HCLTECH', 'DRREDDY', 'CIPLA',
        'DIVISLAB', 'BAJAJFINSV', 'INDUSINDBK', 'JSWSTEEL', 'TATASTEEL'
    ]
    
    trainer = ModelTrainer()
    
    try:
        metrics = await trainer.train_ensemble(training_symbols)
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        if 'ensemble' in metrics and 'validation' in metrics['ensemble']:
            val_metrics = metrics['ensemble']['validation']
            print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")
            print(f"Validation ROC AUC: {val_metrics['roc_auc']:.4f}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
    
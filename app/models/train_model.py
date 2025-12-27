import asyncio
import logging
from datetime import datetime
from app.services.prediction_service import StockPredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def train_model():
    """Train the ML model with Indian stock data"""
    
    logger.info("Starting model training...")
    
    # List of popular Indian stocks for training
    training_symbols = [
        # Large Cap
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
        'ICICIBANK', 'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC',
        'BAJFINANCE', 'LT', 'ASIANPAINT', 'AXISBANK', 'MARUTI',
        
        # Mid Cap
        'ADANIPORTS', 'TATAMOTORS', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO',
        'NESTLEIND', 'POWERGRID', 'NTPC', 'ONGC', 'M&M',
        
        # Additional stocks for diversity
        'WIPRO', 'TECHM', 'HCLTECH', 'DRREDDY', 'CIPLA',
        'DIVISLAB', 'BAJAJFINSV', 'INDUSINDBK', 'JSWSTEEL', 'TATASTEEL'
    ]
    
    prediction_service = StockPredictionService()
    
    try:
        result = await prediction_service.train_model(
            symbols=training_symbols,
            start_date=None,  # Will fetch last 2 years
            end_date=None
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {result}")
        
        # Print metrics
        if 'metrics' in result:
            metrics = result['metrics']
            print("\n" + "="*60)
            print("MODEL PERFORMANCE METRICS")
            print("="*60)
            print(f"Accuracy:       {metrics['accuracy']:.4f}")
            print(f"Precision:      {metrics['precision']:.4f}")
            print(f"Recall:         {metrics['recall']:.4f}")
            print(f"F1 Score:       {metrics['f1_score']:.4f}")
            print(f"ROC AUC:        {metrics['roc_auc']:.4f}")
            print(f"CV F1 Mean:     {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
            print(f"Train Samples:  {metrics['train_samples']}")
            print(f"Test Samples:   {metrics['test_samples']}")
            print(f"Positive Rate:  {metrics['positive_rate']:.4f}")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(train_model())
    
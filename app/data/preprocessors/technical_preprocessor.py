import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalPreprocessor:
    """Preprocess data for technical analysis"""
    
    @staticmethod
    def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data"""
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        # Forward fill missing values (up to 3 days)
        df = df.fillna(method='ffill', limit=3)
        
        # Remove remaining NaN
        df = df.dropna()
        
        # Validate OHLC relationships
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['volume'] >= 0)
        ]
        
        logger.info(f"Cleaned data: {len(df)} valid records")
        
        return df
    
    @staticmethod
    def handle_stock_splits(df: pd.DataFrame) -> pd.DataFrame:
        """Detect and adjust for stock splits"""
        
        # Calculate daily returns
        df['returns'] = df['close'].pct_change()
        
        # Detect potential splits (>25% price change)
        potential_splits = df[abs(df['returns']) > 0.25].index
        
        if len(potential_splits) > 0:
            logger.warning(f"Potential stock splits detected on {len(potential_splits)} dates")
            # Note: Yahoo Finance data is usually pre-adjusted
        
        return df
    
    @staticmethod
    def normalize_volume(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize volume data"""
        
        # Add volume moving average
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # Normalize volume
        df['volume_normalized'] = df['volume'] / df['volume_ma_20']
        
        return df
    
    @staticmethod
    def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Is it month/quarter end?
        df['is_month_end'] = df.index.is_month_end
        df['is_quarter_end'] = df.index.is_quarter_end
        
        return df
    
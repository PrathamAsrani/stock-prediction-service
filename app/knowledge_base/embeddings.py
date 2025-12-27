import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class HistoricalDataCollector:
    """Collect historical stock price data"""
    
    def __init__(self, save_path: str = "data/historical"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def fetch_stock_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "5y"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a stock
        
        Args:
            symbol: Stock ticker
            exchange: NSE or BSE
            start_date: Start date
            end_date: End date
            period: Period if dates not provided (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        try:
            # Convert to Yahoo Finance format
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
            
            logger.info(f"Fetching data for {yahoo_symbol}")
            
            ticker = yf.Ticker(yahoo_symbol)
            
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period=period)
            
            if df.empty:
                logger.warning(f"No data found for {yahoo_symbol}")
                return None
            
            # Clean column names
            df.columns = df.columns.str.lower()
            
            # Add metadata
            df['symbol'] = symbol
            df['exchange'] = exchange
            
            logger.info(f"Fetched {len(df)} records for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        exchange: str = "NSE",
        period: str = "5y"
    ) -> pd.DataFrame:
        """Fetch data for multiple stocks"""
        
        all_data = []
        
        for symbol in symbols:
            df = self.fetch_stock_data(symbol, exchange, period=period)
            if df is not None:
                all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=False)
            logger.info(f"Fetched data for {len(all_data)} stocks")
            return combined
        
        return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save data to CSV"""
        filepath = self.save_path / filename
        df.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
    
    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """Load data from CSV"""
        filepath = self.save_path / filename
        if filepath.exists():
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        return pd.DataFrame()
    
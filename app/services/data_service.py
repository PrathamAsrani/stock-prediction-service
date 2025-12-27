import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from app.schemas.prediction_schema import CandleData

logger = logging.getLogger(__name__)

class DataService:
    """Service for fetching stock data"""
    
    @staticmethod
    def fetch_historical_data(
        symbol: str,
        exchange: str = "NSE",
        days: int = 365
    ) -> Optional[List[CandleData]]:
        """Fetch historical OHLCV data from Yahoo Finance"""
        try:
            # Convert to Yahoo Finance symbol format
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Fetching historical data for {yahoo_symbol} from {start_date.date()} to {end_date.date()}")
            
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data found for {yahoo_symbol}")
                return None
            
            # Convert to CandleData list
            candles = []
            for index, row in df.iterrows():
                candle = CandleData(
                    timestamp=index.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                )
                candles.append(candle)
            
            logger.info(f"Fetched {len(candles)} candles for {yahoo_symbol}")
            
            return candles
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    @staticmethod
    def candles_to_dataframe(candles: List[CandleData]) -> pd.DataFrame:
        """Convert list of candles to DataFrame"""
        df = pd.DataFrame([{
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in candles])
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    @staticmethod
    def get_current_price(symbol: str, exchange: str = "NSE") -> Optional[float]:
        """Get current stock price"""
        try:
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
            ticker = yf.Ticker(yahoo_symbol)
            
            # Get latest price
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            return None
        
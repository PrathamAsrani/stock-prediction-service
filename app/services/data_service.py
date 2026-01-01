import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from app.schemas.prediction_schema import CandleData
from app.config.settings import settings

logger = logging.getLogger(__name__)

class DataService:
    """Service for fetching stock data from Java microservice"""
    
    def __init__(self):
        from app.config.settings import settings
        self.java_service_url = settings.JAVA_SERVICE_URL
        self.timeout = settings.JAVA_SERVICE_TIMEOUT
    
    def fetch_historical_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        days: int = 365
    ) -> Optional[List[CandleData]]:
        """
        Fetch historical OHLCV data from Java microservice
        
        Args:
            symbol: Stock ticker (e.g., "ITC", "RELIANCE")
            exchange: Exchange (NSE/BSE)
            days: Number of days of historical data
            
        Returns:
            List of CandleData objects or None if fetch fails
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates as YYYY-MM-DD
            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")
            
            logger.info(f"Fetching historical data for {symbol} from Java service")
            logger.info(f"Date range: {from_date} to {to_date}")
            
            # Call Java microservice
            url = f"{self.java_service_url}/invest/historical"
            params = {
                "symbol": symbol,
                "exchange": exchange,
                "fromDate": from_date,
                "toDate": to_date,
                "interval": "day",
                "continuous": "true"
            }
            
            logger.debug(f"Calling Java API: {url} with params: {params}")
            
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout
            )
            
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"Java service returned status {response.status_code}: {response.text}")
                return None
            
            # Parse response
            data = response.json()
            
            # Check if the Java service returned success
            if not data.get('success', False):
                logger.error(f"Java service error: {data.get('message', 'Unknown error')}")
                return None
            
            # Extract candles
            candles_data = data.get('candles', [])
            
            if not candles_data:
                logger.warning(f"No candle data found for {symbol}")
                return None
            
            # Convert to CandleData objects
            candles = []
            for candle in candles_data:
                try:
                    # Parse timestamp
                    timestamp_str = candle.get('timestamp')
                    if timestamp_str:
                        # Handle different timestamp formats
                        timestamp = self._parse_timestamp(timestamp_str)
                    else:
                        logger.warning("Candle missing timestamp, skipping")
                        continue
                    
                    candle_obj = CandleData(
                        timestamp=timestamp,
                        open=float(candle.get('open', 0)),
                        high=float(candle.get('high', 0)),
                        low=float(candle.get('low', 0)),
                        close=float(candle.get('close', 0)),
                        volume=int(candle.get('volume', 0))
                    )
                    candles.append(candle_obj)
                    
                except Exception as e:
                    logger.warning(f"Error parsing candle: {e}")
                    continue
            
            logger.info(f"✓ Fetched {len(candles)} candles for {symbol} from Java service")
            
            return candles
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching data from Java service for {symbol}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error to Java service. Is it running at {self.java_service_url}?")
            return None
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from various formats"""
        
        # Try ISO format first (e.g., "2024-12-26T00:00:00")
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            pass
        
        # Try date only format (e.g., "2024-12-26")
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d")
        except:
            pass
        
        # Try with time (e.g., "2024-12-26 00:00:00")
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except:
            pass
        
        # Fallback: just use current time
        logger.warning(f"Could not parse timestamp: {timestamp_str}, using current time")
        return datetime.now()
    
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
    
    def get_current_price(self, symbol: str, exchange: str = "NSE") -> Optional[float]:
        """Get current stock price from Java service"""
        try:
            url = f"{self.java_service_url}/invest/latest"
            params = {
                "symbol": symbol,
                "exchange": exchange
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get('close', 0))
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            return None
    
    def check_java_service_health(self) -> bool:
        """Check if Java microservice is reachable"""
        try:
            # Try to hit the health endpoint or any simple endpoint
            response = requests.get(
                f"{self.java_service_url}/invest/cache/clear",
                timeout=5
            )
            return response.status_code in [200, 404]  # Even 404 means service is up
        except:
            return False
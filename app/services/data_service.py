import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from app.schemas.prediction_schema import CandleData
from app.config.settings import settings

logger = logging.getLogger(__name__)

class DataService:
    """Service for fetching stock data with fallback to hardcoded data"""
    
    def __init__(self):
        from app.config.settings import settings
        self.java_service_url = settings.JAVA_SERVICE_URL
        self.timeout = settings.JAVA_SERVICE_TIMEOUT
        
        # Hardcoded stock data for fallback
        self.fallback_data = self._initialize_fallback_data()
    
    def _initialize_fallback_data(self):
        """Initialize hardcoded stock data for common symbols"""
        return {
            'PGHL': self._get_pghl_data(),
            'ITC': self._get_itc_sample_data(),
            # Add more stocks as needed
        }
    
    def _get_pghl_data(self) -> List[CandleData]:
        """PGHL stock data from Screener.in graph"""
        
        # Data extracted from the graph you provided
        pghl_price_data = [
            ["2025-01-02", "5252.95"],
            ["2025-01-03", "5203.95"],
            ["2025-01-06", "5081.70"],
            ["2025-01-07", "5119.05"],
            ["2025-01-08", "5064.35"],
            ["2025-01-09", "5071.35"],
            ["2025-01-10", "5258.45"],
            ["2025-01-13", "5199.60"],
            ["2025-01-14", "5351.10"],
            ["2025-01-15", "5430.70"],
            ["2025-01-16", "5382.80"],
            ["2025-01-17", "5411.05"],
            ["2025-01-20", "5488.80"],
            ["2025-01-21", "5484.05"],
            ["2025-01-22", "5408.15"],
            ["2025-01-23", "5511.65"],
            ["2025-01-24", "5406.75"],
            ["2025-01-27", "5275.10"],
            ["2025-01-28", "5242.50"],
            ["2025-01-29", "5254.30"],
            ["2025-01-30", "5290.70"],
            ["2025-01-31", "5382.60"],
            ["2025-02-01", "5393.95"],
            ["2025-02-03", "5217.80"],
            ["2025-02-04", "5302.40"],
            ["2025-02-05", "5370.70"],
            ["2025-02-06", "5413.40"],
            ["2025-02-07", "5434.05"],
            ["2025-02-10", "5330.30"],
            ["2025-02-11", "5261.60"],
            ["2025-02-12", "5365.15"],
            ["2025-02-13", "5308.30"],
            ["2025-02-14", "5237.50"],
            ["2025-02-17", "5301.95"],
            ["2025-02-18", "5307.85"],
            ["2025-02-19", "5301.80"],
            ["2025-02-20", "5345.60"],
            ["2025-02-21", "5275.60"],
            ["2025-02-24", "5171.55"],
            ["2025-02-25", "5356.45"],
            ["2025-02-27", "5341.20"],
            ["2025-02-28", "5034.05"],
            ["2025-03-03", "5160.15"],
            ["2025-03-04", "5100.35"],
            ["2025-03-05", "5122.50"],
            ["2025-03-06", "5213.25"],
            ["2025-03-07", "5260.25"],
            ["2025-03-10", "5264.75"],
            ["2025-03-11", "5284.25"],
            ["2025-03-12", "5275.10"],
            ["2025-03-13", "5239.90"],
            ["2025-03-17", "5151.75"],
            ["2025-03-18", "5209.65"],
            ["2025-03-19", "5232.50"],
            ["2025-03-20", "5315.75"],
            ["2025-03-21", "5503.35"],
            ["2025-03-24", "5461.45"],
            ["2025-03-25", "5422.75"],
            ["2025-03-26", "5311.80"],
            ["2025-03-27", "5002.60"],
            ["2025-03-28", "5146.50"],
            ["2025-04-01", "5154.90"],
            ["2025-04-02", "5256.55"],
            ["2025-04-03", "5301.10"],
            ["2025-04-04", "5222.25"],
            ["2025-04-07", "5148.20"],
            ["2025-04-08", "5113.55"],
            ["2025-04-09", "5221.35"],
            ["2025-04-11", "5167.30"],
            ["2025-04-15", "5230.50"],
            ["2025-04-16", "5199.50"],
            ["2025-04-17", "5218.00"],
            ["2025-04-21", "5223.50"],
            ["2025-04-22", "5168.50"],
            ["2025-04-23", "5240.00"],
            ["2025-04-24", "5221.50"],
            ["2025-04-25", "5063.50"],
            ["2025-04-28", "5103.00"],
            ["2025-04-29", "5111.50"],
            ["2025-04-30", "5082.00"],
            ["2025-05-02", "5063.50"],
            ["2025-05-05", "5020.00"],
            ["2025-05-06", "5035.00"],
            ["2025-05-07", "5024.50"],
            ["2025-05-08", "5024.00"],
            ["2025-05-09", "5039.50"],
            ["2025-05-12", "5079.00"],
            ["2025-05-13", "5082.50"],
            ["2025-05-14", "5231.50"],
            ["2025-05-15", "5244.00"],
            ["2025-05-16", "5195.00"],
            ["2025-05-19", "5250.50"],
            ["2025-05-20", "5284.50"],
            ["2025-05-21", "5373.50"],
            ["2025-05-22", "5401.00"],
            ["2025-05-23", "5400.50"],
            ["2025-05-26", "5399.00"],
            ["2025-05-27", "5528.00"],
            ["2025-05-28", "5729.50"],
            ["2025-05-29", "5834.50"],
            ["2025-05-30", "5748.00"],
            ["2025-06-02", "5757.00"],
            ["2025-06-03", "5749.50"],
            ["2025-06-04", "5604.00"],
            ["2025-06-05", "5579.00"],
            ["2025-06-06", "5560.50"],
            ["2025-06-09", "5683.50"],
            ["2025-06-10", "5809.00"],
            ["2025-06-11", "5864.50"],
            ["2025-06-12", "5839.50"],
            ["2025-06-13", "5831.00"],
            ["2025-06-16", "5908.00"],
            ["2025-06-17", "5911.00"],
            ["2025-06-18", "5951.00"],
            ["2025-06-19", "5874.50"],
            ["2025-06-20", "5907.00"],
            ["2025-06-23", "5921.50"],
            ["2025-06-24", "5915.00"],
            ["2025-06-25", "5903.50"],
            ["2025-06-26", "5855.50"],
            ["2025-06-27", "5921.00"],
            ["2025-06-30", "5880.50"],
            ["2025-07-01", "5823.00"],
            ["2025-07-02", "5844.00"],
            ["2025-07-03", "5854.00"],
            ["2025-07-04", "5756.50"],
            ["2025-07-07", "5790.00"],
            ["2025-07-08", "5707.00"],
            ["2025-07-09", "5693.00"],
            ["2025-07-10", "5677.50"],
            ["2025-07-11", "5755.50"],
            ["2025-07-14", "5816.00"],
            ["2025-07-15", "5859.00"],
            ["2025-07-16", "5854.00"],
            ["2025-07-17", "5874.00"],
            ["2025-07-18", "5862.50"],
            ["2025-07-21", "5875.00"],
            ["2025-07-22", "5865.00"],
            ["2025-07-23", "5932.50"],
            ["2025-07-24", "5828.00"],
            ["2025-07-25", "5810.50"],
            ["2025-07-28", "5851.50"],
            ["2025-07-29", "5935.00"],
            ["2025-07-30", "5947.00"],
            ["2025-07-31", "5895.50"],
            ["2025-08-01", "6217.00"],
            ["2025-08-04", "6284.50"],
            ["2025-08-05", "6173.00"],
            ["2025-08-06", "6152.50"],
            ["2025-08-07", "6313.00"],
            ["2025-08-08", "6343.00"],
            ["2025-08-11", "6338.50"],
            ["2025-08-12", "6336.00"],
            ["2025-08-13", "6182.50"],
            ["2025-08-14", "6295.00"],
            ["2025-08-18", "6223.50"],
            ["2025-08-19", "6291.00"],
            ["2025-08-20", "6285.50"],
            ["2025-08-21", "6338.50"],
            ["2025-08-22", "6581.50"],
            ["2025-08-25", "6414.00"],
            ["2025-08-26", "6310.00"],
            ["2025-08-28", "6298.50"],
            ["2025-08-29", "6370.00"],
            ["2025-09-01", "6311.00"],
            ["2025-09-02", "6332.00"],
            ["2025-09-03", "6316.50"],
            ["2025-09-04", "6462.00"],
            ["2025-09-05", "6465.00"],
            ["2025-09-08", "6528.00"],
            ["2025-09-09", "6475.00"],
            ["2025-09-10", "6329.50"],
            ["2025-09-11", "6332.00"],
            ["2025-09-12", "6243.50"],
            ["2025-09-15", "6271.00"],
            ["2025-09-16", "6353.50"],
            ["2025-09-17", "6356.50"],
            ["2025-09-18", "6265.50"],
            ["2025-09-19", "6366.50"],
            ["2025-09-22", "6251.00"],
            ["2025-09-23", "6290.00"],
            ["2025-09-24", "6289.50"],
            ["2025-09-25", "6368.00"],
            ["2025-09-26", "6288.50"],
            ["2025-09-29", "6301.50"],
            ["2025-09-30", "6307.50"],
            ["2025-10-01", "6302.00"],
            ["2025-10-03", "6271.50"],
            ["2025-10-06", "6201.00"],
            ["2025-10-07", "6234.50"],
            ["2025-10-08", "6234.50"],
            ["2025-10-09", "6072.50"],
            ["2025-10-10", "6199.00"],
            ["2025-10-13", "6118.50"],
            ["2025-10-14", "6131.50"],
            ["2025-10-15", "6173.00"],
            ["2025-10-16", "6284.50"],
            ["2025-10-17", "6244.00"],
            ["2025-10-20", "6145.00"],
            ["2025-10-21", "6172.00"],
            ["2025-10-23", "6116.00"],
            ["2025-10-24", "6083.50"],
            ["2025-10-27", "6160.50"],
            ["2025-10-28", "6183.00"],
            ["2025-10-29", "6190.00"],
            ["2025-10-30", "6176.50"],
            ["2025-10-31", "6166.50"],
            ["2025-11-03", "6085.50"],
            ["2025-11-04", "6066.00"],
            ["2025-11-06", "5996.50"],
            ["2025-11-07", "5839.00"],
            ["2025-11-10", "5941.00"],
            ["2025-11-11", "5947.00"],
            ["2025-11-12", "5896.50"],
            ["2025-11-13", "5854.50"],
            ["2025-11-14", "5890.00"],
            ["2025-11-17", "5915.50"],
            ["2025-11-18", "5890.50"],
            ["2025-11-19", "5892.50"],
            ["2025-11-20", "5906.00"],
            ["2025-11-21", "5830.50"],
            ["2025-11-24", "5867.00"],
            ["2025-11-25", "5825.50"],
            ["2025-11-26", "5815.50"],
            ["2025-11-27", "5775.50"],
            ["2025-11-28", "5722.50"],
            ["2025-12-01", "5817.50"],
            ["2025-12-02", "5780.50"],
            ["2025-12-03", "5833.00"],
            ["2025-12-04", "5854.00"],
            ["2025-12-05", "5713.00"],
            ["2025-12-08", "5639.00"],
            ["2025-12-09", "5623.50"],
            ["2025-12-10", "5674.50"],
            ["2025-12-11", "5656.00"],
            ["2025-12-12", "5577.50"],
            ["2025-12-15", "5665.50"],
            ["2025-12-16", "5685.00"],
            ["2025-12-17", "5708.50"],
            ["2025-12-18", "5650.00"],
            ["2025-12-19", "5662.00"],
            ["2025-12-22", "5640.00"],
            ["2025-12-23", "5670.00"],
            ["2025-12-24", "5641.00"],
            ["2025-12-26", "5644.50"],
            ["2025-12-29", "5680.50"],
            ["2025-12-30", "5651.50"],
            ["2025-12-31", "5703.00"],
            ["2026-01-01", "5722.50"],
            ["2026-01-02", "5662.00"]
        ]
        
        candles = []
        
        for date_str, close_str in pghl_price_data:
            try:
                timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                close = float(close_str)
                
                # Generate realistic OHLC from close price
                # Typical daily range is 1-3%
                volatility = 0.015  # 1.5% typical volatility
                high = close * (1 + volatility)
                low = close * (1 - volatility)
                
                # Open is close to previous close (or current close for first)
                open_price = close * (1 + (volatility * 0.3 * (1 if len(candles) % 2 == 0 else -1)))
                
                # Realistic volume (based on typical PGHL volumes)
                base_volume = 5000
                volume_variance = int(base_volume * 2 * (timestamp.day % 10) / 10)
                volume = base_volume + volume_variance
                
                candle = CandleData(
                    timestamp=timestamp,
                    open=round(open_price, 2),
                    high=round(high, 2),
                    low=round(low, 2),
                    close=close,
                    volume=volume
                )
                candles.append(candle)
                
            except Exception as e:
                logger.warning(f"Error parsing PGHL data: {e}")
                continue
        
        return candles
    
    def _get_itc_sample_data(self) -> List[CandleData]:
        """Sample ITC data for fallback"""
        # You can add ITC data here similarly
        candles = []
        base_price = 465.0
        start_date = datetime(2025, 1, 1)
        
        for i in range(365):
            current_date = start_date + timedelta(days=i)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Simulate price movement
            price_change = (i % 10 - 5) * 0.5
            close = base_price + price_change
            
            candle = CandleData(
                timestamp=current_date,
                open=close * 0.998,
                high=close * 1.015,
                low=close * 0.985,
                close=close,
                volume=1000000 + (i * 1000)
            )
            candles.append(candle)
        
        return candles
    
    def fetch_historical_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        days: int = 365
    ) -> Optional[List[CandleData]]:
        """
        Fetch historical OHLCV data with fallback to hardcoded data
        
        Priority:
        1. Java microservice
        2. Hardcoded fallback data
        """
        
        # Try Java service first
        logger.info(f"Attempting to fetch {symbol} from Java service")
        java_candles = self._fetch_from_java_service(symbol, exchange, days)
        
        if java_candles:
            logger.info(f"✓ Fetched {len(java_candles)} candles from Java service")
            return java_candles
        
        # Fallback to hardcoded data
        logger.info(f"Java service unavailable. Using fallback data for {symbol}")
        
        if symbol in self.fallback_data:
            candles = self.fallback_data[symbol]
            
            # Filter to requested date range
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_candles = [c for c in candles if c.timestamp >= cutoff_date]
            
            logger.info(f"✓ Using {len(filtered_candles)} hardcoded candles for {symbol}")
            return filtered_candles
        
        logger.error(f"No fallback data available for {symbol}")
        return None
    
    def _fetch_from_java_service(
        self, 
        symbol: str, 
        exchange: str, 
        days: int
    ) -> Optional[List[CandleData]]:
        """Fetch from Java microservice"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")
            
            url = f"{self.java_service_url}/invest/historical"
            params = {
                "symbol": symbol,
                "exchange": exchange,
                "fromDate": from_date,
                "toDate": to_date,
                "interval": "day",
                "continuous": "true"
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.warning(f"Java service returned {response.status_code}")
                return None
            
            data = response.json()
            
            if not data.get('success', False):
                logger.warning(f"Java service error: {data.get('message', 'Unknown')}")
                return None
            
            candles_data = data.get('candles', [])
            if not candles_data:
                return None
            
            candles = []
            for candle in candles_data:
                try:
                    timestamp_str = candle.get('timestamp')
                    if timestamp_str:
                        timestamp = self._parse_timestamp(timestamp_str)
                    else:
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
            
            return candles if candles else None
            
        except Exception as e:
            logger.warning(f"Java service error: {str(e)}")
            return None
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from various formats"""
        
        # Try ISO format first
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            pass
        
        # Try date only format
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d")
        except:
            pass
        
        # Try with time
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except:
            pass
        
        # Fallback
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
        """Get current stock price"""
        
        # Try Java service
        try:
            url = f"{self.java_service_url}/invest/latest"
            params = {"symbol": symbol, "exchange": exchange}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data.get('close', 0))
        except:
            pass
        
        # Fallback to hardcoded data
        if symbol in self.fallback_data:
            candles = self.fallback_data[symbol]
            if candles:
                return candles[-1].close
        
        return None
    
    def check_java_service_health(self) -> bool:
        """Check if Java microservice is reachable"""
        try:
            response = requests.get(f"{self.java_service_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
            
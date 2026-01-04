import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import json

from app.schemas.prediction_schema import CandleData

logger = logging.getLogger(__name__)

class DataService:
    """Enhanced data service with Yahoo Finance and dynamic JSON fallback"""
    
    def __init__(self):
        self.java_service_url = "http://localhost:3000"
        self.fallback_data = {}  # Initialize empty dictionary
        self._load_custom_stocks()  # Load stocks from JSON
    
    def _initialize_fallback_data(self):
        """Initialize hardcoded stock data for common symbols"""
        return {
            'PGHL': self._get_pghl_data(),
            'ITC': self._get_itc_sample_data(),
            # Add more stocks dynamically using add_stock_from_json()
        }
    
    def add_stock_from_json(self, symbol: str, json_string: str, json_path: str = "datasets.0.values"):
        """
        ✨ DYNAMIC FUNCTION: Add a new stock to fallback data from JSON
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            json_string: JSON string from Screener.in or similar source
            json_path: Path to price data in JSON (default: "datasets.0.values")
                      For Screener.in format: "datasets.0.values"
                      For other formats, adjust accordingly
        
        Example usage:
            json_data = '''
            {
                "datasets": [
                    {
                        "metric": "Price",
                        "values": [
                            ["2025-01-02", "2500.50"],
                            ["2025-01-03", "2520.75"],
                            ...
                        ]
                    }
                ]
            }
            '''
            
            data_service.add_stock_from_json('RELIANCE', json_data)
        """
        try:
            # Parse JSON
            data = json.loads(json_string)
            
            # Navigate to price data using json_path
            price_data = self._get_nested_value(data, json_path)
            
            if not price_data:
                logger.error(f"Could not find price data at path: {json_path}")
                return False
            
            # Generate candles from price data
            candles = self._generate_candles_from_price_data(symbol, price_data)
            
            if candles:
                # Add to fallback data
                self.fallback_data[symbol] = candles
                logger.info(f"✓ Added {len(candles)} candles for {symbol} to fallback data")
                return True
            else:
                logger.error(f"Failed to generate candles for {symbol}")
                return False
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return False
        except Exception as e:
            logger.error(f"Error adding {symbol}: {e}")
            return False
    
    def _get_nested_value(self, data: dict, path: str):
        """
        Navigate nested dictionary using dot notation
        
        Example: "datasets.0.values" → data['datasets'][0]['values']
        """
        keys = path.split('.')
        current = data
        
        for key in keys:
            if key.isdigit():
                # Array index
                current = current[int(key)]
            else:
                # Dictionary key
                current = current.get(key)
            
            if current is None:
                return None
        
        return current
    
    def _generate_candles_from_price_data(
        self, 
        symbol: str, 
        price_data: List[List],
        volatility: float = 0.015
    ) -> List[CandleData]:
        """
        Generate realistic OHLCV candles from [date, close] price data
        
        Args:
            symbol: Stock symbol
            price_data: List of [date_string, close_price] pairs
            volatility: Daily volatility % (default 1.5%)
        
        Returns:
            List of CandleData objects
        """
        candles = []
        
        for i, price_entry in enumerate(price_data):
            try:
                # Handle different formats
                if len(price_entry) >= 2:
                    date_str = price_entry[0]
                    close_str = price_entry[1]
                else:
                    logger.warning(f"Invalid price entry format: {price_entry}")
                    continue
                
                # Parse date
                timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                close = float(close_str)
                
                # Generate realistic OHLC from close price
                # Typical daily range is 1-3% (configurable via volatility)
                high = close * (1 + volatility)
                low = close * (1 - volatility)
                
                # Open is close to previous close (or current close for first)
                open_price = close * (1 + (volatility * 0.3 * (1 if i % 2 == 0 else -1)))
                
                # Realistic volume (varies by day)
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
                logger.warning(f"Error parsing price data for {symbol}: {e}")
                continue
        
        return candles
    
    # ========================================================================
    # 📝 PASTE YOUR JSON DATA HERE
    # ========================================================================
    
    def _load_custom_stocks(self):
        """
        🎯 PASTE YOUR STOCK JSON DATA HERE
        
        Example:
            RELIANCE_JSON = '''
            {
                "datasets": [
                    {
                        "metric": "Price",
                        "values": [
                            ["2025-01-02", "2500.50"],
                            ["2025-01-03", "2520.75"],
                            ...
                        ]
                    }
                ]
            }
            '''
            
            self.add_stock_from_json('RELIANCE', RELIANCE_JSON)
        
        Add as many stocks as you want!
        """
        
        # Example: RELIANCE stock (uncomment and add your JSON)
        # RELIANCE_JSON = '''
        # {
        #     "datasets": [
        #         {
        #             "metric": "Price",
        #             "values": [
        #                 ["2025-01-02", "2500.50"],
        #                 ["2025-01-03", "2520.75"]
        #             ]
        #         }
        #     ]
        # }
        # '''
        # self.add_stock_from_json('RELIANCE', RELIANCE_JSON)
        
        # Example: TCS stock
        # TCS_JSON = '''{ ... }'''
        # self.add_stock_from_json('TCS', TCS_JSON)

        WAAREEINDO_JSON = '''
        {
            "datasets": [
                {
                    "metric": "Price",
                    "label": "Price on NSE",
                    "values": [
                        [
                            "2025-06-19",
                            "173.32"
                        ],
                        [
                            "2025-06-20",
                            "181.98"
                        ],
                        [
                            "2025-06-23",
                            "191.07"
                        ],
                        [
                            "2025-06-24",
                            "200.62"
                        ],
                        [
                            "2025-06-25",
                            "210.65"
                        ],
                        [
                            "2025-06-26",
                            "221.18"
                        ],
                        [
                            "2025-06-27",
                            "232.23"
                        ],
                        [
                            "2025-07-07",
                            "243.84"
                        ],
                        [
                            "2025-07-14",
                            "256.03"
                        ],
                        [
                            "2025-07-21",
                            "268.83"
                        ],
                        [
                            "2025-07-28",
                            "282.27"
                        ],
                        [
                            "2025-08-01",
                            "296.40"
                        ],
                        [
                            "2025-08-04",
                            "311.20"
                        ],
                        [
                            "2025-08-05",
                            "326.75"
                        ],
                        [
                            "2025-08-06",
                            "343.05"
                        ],
                        [
                            "2025-08-07",
                            "360.20"
                        ],
                        [
                            "2025-08-08",
                            "378.20"
                        ],
                        [
                            "2025-08-18",
                            "397.10"
                        ],
                        [
                            "2025-08-25",
                            "416.95"
                        ],
                        [
                            "2025-09-01",
                            "437.75"
                        ],
                        [
                            "2025-09-08",
                            "459.60"
                        ],
                        [
                            "2025-09-15",
                            "482.55"
                        ],
                        [
                            "2025-09-22",
                            "506.65"
                        ],
                        [
                            "2025-09-29",
                            "531.95"
                        ],
                        [
                            "2025-10-01",
                            "558.50"
                        ],
                        [
                            "2025-10-03",
                            "586.40"
                        ],
                        [
                            "2025-10-06",
                            "615.70"
                        ],
                        [
                            "2025-10-07",
                            "646.45"
                        ],
                        [
                            "2025-10-08",
                            "678.75"
                        ],
                        [
                            "2025-10-09",
                            "711.70"
                        ],
                        [
                            "2025-10-13",
                            "676.15"
                        ],
                        [
                            "2025-10-20",
                            "642.35"
                        ],
                        [
                            "2025-10-27",
                            "610.25"
                        ],
                        [
                            "2025-11-03",
                            "579.75"
                        ],
                        [
                            "2025-11-10",
                            "550.80"
                        ],
                        [
                            "2025-11-17",
                            "556.90"
                        ],
                        [
                            "2025-11-24",
                            "529.05"
                        ],
                        [
                            "2025-12-01",
                            "555.50"
                        ],
                        [
                            "2025-12-02",
                            "583.25"
                        ],
                        [
                            "2025-12-03",
                            "554.10"
                        ],
                        [
                            "2025-12-04",
                            "526.40"
                        ],
                        [
                            "2025-12-05",
                            "500.10"
                        ],
                        [
                            "2025-12-08",
                            "475.10"
                        ],
                        [
                            "2025-12-09",
                            "451.35"
                        ],
                        [
                            "2025-12-10",
                            "455.15"
                        ],
                        [
                            "2025-12-11",
                            "452.20"
                        ],
                        [
                            "2025-12-12",
                            "440.05"
                        ],
                        [
                            "2025-12-15",
                            "433.90"
                        ],
                        [
                            "2025-12-16",
                            "424.70"
                        ],
                        [
                            "2025-12-17",
                            "403.50"
                        ],
                        [
                            "2025-12-18",
                            "417.10"
                        ],
                        [
                            "2025-12-19",
                            "427.55"
                        ],
                        [
                            "2025-12-22",
                            "448.90"
                        ],
                        [
                            "2025-12-23",
                            "471.30"
                        ],
                        [
                            "2025-12-24",
                            "494.85"
                        ],
                        [
                            "2025-12-26",
                            "519.55"
                        ],
                        [
                            "2025-12-29",
                            "542.15"
                        ],
                        [
                            "2026-01-01",
                            "536.50"
                        ],
                        [
                            "2026-01-02",
                            "512.60"
                        ]
                    ],
                    "meta": {
                        "is_weekly": false
                    }
                },
                {
                    "metric": "DMA50",
                    "label": "50 DMA",
                    "values": [
                        [
                            "2025-06-19",
                            "10.67"
                        ],
                        [
                            "2025-06-20",
                            "17.39"
                        ],
                        [
                            "2025-06-23",
                            "24.20"
                        ],
                        [
                            "2025-06-24",
                            "31.12"
                        ],
                        [
                            "2025-06-25",
                            "38.16"
                        ],
                        [
                            "2025-06-26",
                            "45.34"
                        ],
                        [
                            "2025-06-27",
                            "52.67"
                        ],
                        [
                            "2025-07-07",
                            "60.16"
                        ],
                        [
                            "2025-07-14",
                            "67.85"
                        ],
                        [
                            "2025-07-21",
                            "75.73"
                        ],
                        [
                            "2025-07-28",
                            "83.83"
                        ],
                        [
                            "2025-08-01",
                            "92.16"
                        ],
                        [
                            "2025-08-04",
                            "100.75"
                        ],
                        [
                            "2025-08-05",
                            "109.62"
                        ],
                        [
                            "2025-08-06",
                            "118.77"
                        ],
                        [
                            "2025-08-07",
                            "128.24"
                        ],
                        [
                            "2025-08-08",
                            "138.04"
                        ],
                        [
                            "2025-08-18",
                            "148.20"
                        ],
                        [
                            "2025-08-25",
                            "158.74"
                        ],
                        [
                            "2025-09-01",
                            "169.68"
                        ],
                        [
                            "2025-09-08",
                            "181.05"
                        ],
                        [
                            "2025-09-15",
                            "192.87"
                        ],
                        [
                            "2025-09-22",
                            "205.18"
                        ],
                        [
                            "2025-09-29",
                            "217.99"
                        ],
                        [
                            "2025-10-01",
                            "231.35"
                        ],
                        [
                            "2025-10-03",
                            "245.27"
                        ],
                        [
                            "2025-10-06",
                            "259.80"
                        ],
                        [
                            "2025-10-07",
                            "274.96"
                        ],
                        [
                            "2025-10-08",
                            "290.79"
                        ],
                        [
                            "2025-10-09",
                            "307.30"
                        ],
                        [
                            "2025-10-13",
                            "321.77"
                        ],
                        [
                            "2025-10-20",
                            "334.34"
                        ],
                        [
                            "2025-10-27",
                            "345.16"
                        ],
                        [
                            "2025-11-03",
                            "354.36"
                        ],
                        [
                            "2025-11-10",
                            "362.06"
                        ],
                        [
                            "2025-11-17",
                            "369.70"
                        ],
                        [
                            "2025-11-24",
                            "375.95"
                        ],
                        [
                            "2025-12-01",
                            "382.99"
                        ],
                        [
                            "2025-12-02",
                            "390.84"
                        ],
                        [
                            "2025-12-03",
                            "397.25"
                        ],
                        [
                            "2025-12-04",
                            "402.31"
                        ],
                        [
                            "2025-12-05",
                            "406.15"
                        ],
                        [
                            "2025-12-08",
                            "408.85"
                        ],
                        [
                            "2025-12-09",
                            "410.52"
                        ],
                        [
                            "2025-12-10",
                            "412.27"
                        ],
                        [
                            "2025-12-11",
                            "413.83"
                        ],
                        [
                            "2025-12-12",
                            "414.86"
                        ],
                        [
                            "2025-12-15",
                            "415.61"
                        ],
                        [
                            "2025-12-16",
                            "415.96"
                        ],
                        [
                            "2025-12-17",
                            "415.48"
                        ],
                        [
                            "2025-12-18",
                            "415.54"
                        ],
                        [
                            "2025-12-19",
                            "416.01"
                        ],
                        [
                            "2025-12-22",
                            "417.30"
                        ],
                        [
                            "2025-12-23",
                            "419.42"
                        ],
                        [
                            "2025-12-24",
                            "422.38"
                        ],
                        [
                            "2025-12-26",
                            "426.19"
                        ],
                        [
                            "2025-12-29",
                            "430.73"
                        ],
                        [
                            "2026-01-01",
                            "434.88"
                        ],
                        [
                            "2026-01-02",
                            "437.93"
                        ]
                    ],
                    "meta": {}
                },
                {
                    "metric": "DMA200",
                    "label": "200 DMA",
                    "values": [
                        [
                            "2025-06-19",
                            "5.41"
                        ],
                        [
                            "2025-06-20",
                            "7.17"
                        ],
                        [
                            "2025-06-23",
                            "9.00"
                        ],
                        [
                            "2025-06-24",
                            "10.91"
                        ],
                        [
                            "2025-06-25",
                            "12.89"
                        ],
                        [
                            "2025-06-26",
                            "14.97"
                        ],
                        [
                            "2025-06-27",
                            "17.13"
                        ],
                        [
                            "2025-07-07",
                            "19.38"
                        ],
                        [
                            "2025-07-14",
                            "21.74"
                        ],
                        [
                            "2025-07-21",
                            "24.20"
                        ],
                        [
                            "2025-07-28",
                            "26.77"
                        ],
                        [
                            "2025-08-01",
                            "29.45"
                        ],
                        [
                            "2025-08-04",
                            "32.25"
                        ],
                        [
                            "2025-08-05",
                            "35.18"
                        ],
                        [
                            "2025-08-06",
                            "38.25"
                        ],
                        [
                            "2025-08-07",
                            "41.45"
                        ],
                        [
                            "2025-08-08",
                            "44.80"
                        ],
                        [
                            "2025-08-18",
                            "48.31"
                        ],
                        [
                            "2025-08-25",
                            "51.97"
                        ],
                        [
                            "2025-09-01",
                            "55.81"
                        ],
                        [
                            "2025-09-08",
                            "59.83"
                        ],
                        [
                            "2025-09-15",
                            "64.04"
                        ],
                        [
                            "2025-09-22",
                            "68.44"
                        ],
                        [
                            "2025-09-29",
                            "73.05"
                        ],
                        [
                            "2025-10-01",
                            "77.88"
                        ],
                        [
                            "2025-10-03",
                            "82.94"
                        ],
                        [
                            "2025-10-06",
                            "88.24"
                        ],
                        [
                            "2025-10-07",
                            "93.80"
                        ],
                        [
                            "2025-10-08",
                            "99.62"
                        ],
                        [
                            "2025-10-09",
                            "105.71"
                        ],
                        [
                            "2025-10-13",
                            "111.38"
                        ],
                        [
                            "2025-10-20",
                            "116.67"
                        ],
                        [
                            "2025-10-27",
                            "121.58"
                        ],
                        [
                            "2025-11-03",
                            "126.14"
                        ],
                        [
                            "2025-11-10",
                            "130.36"
                        ],
                        [
                            "2025-11-17",
                            "134.61"
                        ],
                        [
                            "2025-11-24",
                            "138.53"
                        ],
                        [
                            "2025-12-01",
                            "142.68"
                        ],
                        [
                            "2025-12-02",
                            "147.07"
                        ],
                        [
                            "2025-12-03",
                            "151.12"
                        ],
                        [
                            "2025-12-04",
                            "154.85"
                        ],
                        [
                            "2025-12-05",
                            "158.28"
                        ],
                        [
                            "2025-12-08",
                            "161.44"
                        ],
                        [
                            "2025-12-09",
                            "164.32"
                        ],
                        [
                            "2025-12-10",
                            "167.22"
                        ],
                        [
                            "2025-12-11",
                            "170.05"
                        ],
                        [
                            "2025-12-12",
                            "172.74"
                        ],
                        [
                            "2025-12-15",
                            "175.34"
                        ],
                        [
                            "2025-12-16",
                            "177.82"
                        ],
                        [
                            "2025-12-17",
                            "180.06"
                        ],
                        [
                            "2025-12-18",
                            "182.42"
                        ],
                        [
                            "2025-12-19",
                            "184.86"
                        ],
                        [
                            "2025-12-22",
                            "187.49"
                        ],
                        [
                            "2025-12-23",
                            "190.31"
                        ],
                        [
                            "2025-12-24",
                            "193.34"
                        ],
                        [
                            "2025-12-26",
                            "196.59"
                        ],
                        [
                            "2025-12-29",
                            "200.03"
                        ],
                        [
                            "2026-01-01",
                            "203.37"
                        ],
                        [
                            "2026-01-02",
                            "206.45"
                        ]
                    ],
                    "meta": {}
                },
                {
                    "metric": "Volume",
                    "label": "Volume",
                    "values": [
                        [
                            "2025-06-19",
                            7986,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-06-20",
                            3680,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-06-23",
                            4974,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-06-24",
                            1419,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-06-25",
                            3141,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-06-26",
                            860,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-06-27",
                            1433,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-07-07",
                            4529,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-07-14",
                            2765,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-07-21",
                            15656,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-07-28",
                            2437,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-08-01",
                            1650,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-08-04",
                            2423,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-08-05",
                            1018,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-08-06",
                            1052,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-08-07",
                            1091,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-08-08",
                            1464,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-08-18",
                            2893,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-08-25",
                            3754,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-09-01",
                            4485,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-09-08",
                            2783,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-09-15",
                            2552,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-09-22",
                            3180,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-09-29",
                            2684,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-10-01",
                            1298,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-10-03",
                            1211,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-10-06",
                            2554,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-10-07",
                            957933,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-10-08",
                            122771,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-10-09",
                            1296987,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-10-13",
                            612412,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-10-20",
                            168648,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-10-27",
                            287752,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-11-03",
                            54585,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-11-10",
                            262663,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-11-17",
                            161090,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-11-24",
                            170885,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-01",
                            33586,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-02",
                            15353,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-03",
                            384555,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-04",
                            119211,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-05",
                            187252,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-08",
                            66415,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-09",
                            27290,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-10",
                            245874,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-11",
                            137188,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-12",
                            59111,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-15",
                            92603,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-16",
                            44216,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-17",
                            114661,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-18",
                            154354,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-19",
                            74721,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-22",
                            22508,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-23",
                            118532,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-24",
                            175947,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-26",
                            148235,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2025-12-29",
                            286496,
                            {
                                "delivery": null
                            }
                        ],
                        [
                            "2026-01-01",
                            606548,
                            {
                                "delivery": 42
                            }
                        ],
                        [
                            "2026-01-02",
                            147771,
                            {
                                "delivery": 65
                            }
                        ]
                    ],
                    "meta": {}
                }
            ]
        }
        '''
        self.add_stock_from_json('WAAREEINDO', WAAREEINDO_JSON)
        pass
    
    # ========================================================================
    # Original Methods (PGHL, ITC, etc.)
    # ========================================================================
    
    def _get_pghl_data(self) -> List[CandleData]:
        """
        Placeholder - PGHL data now loaded via _load_custom_stocks() JSON
        This method kept for backwards compatibility
        """
        return []
    
    def _get_itc_sample_data(self) -> List[CandleData]:
        """Sample ITC data for fallback"""
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
    
    # ========================================================================
    # Data Fetching Logic
    # ========================================================================
    
    def fetch_historical_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        days: int = 365,
        min_candles: int = 50
    ) -> Optional[List[CandleData]]:
        """
        Fetch historical OHLCV data with intelligent multi-source fallback
        
        Priority:
        1. Java microservice (Zerodha)
        2. Check if fallback has sufficient data (≥50 candles)
           - If YES: Use fallback
           - If NO: Try Yahoo Finance first
        3. Yahoo Finance (FREE!)
        4. Hardcoded fallback (as last resort)
        
        Args:
            symbol: Stock symbol
            exchange: Exchange (NSE, BSE)
            days: Number of days of historical data
            min_candles: Minimum candles needed for technical analysis (default: 50)
        """
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Try Java microservice first
        logger.info(f"Attempting to fetch {symbol} from Java service")
        candles = self._fetch_from_java_service(symbol, exchange, start_date, end_date)
        
        if candles and len(candles) >= min_candles:
            logger.info(f"✓ Got {len(candles)} candles from Java service")
            return candles
        
        # Check if fallback data exists and has sufficient candles
        if symbol in self.fallback_data:
            fallback_candles = self.fallback_data[symbol]
            # Filter by date range
            filtered_fallback = [c for c in fallback_candles if start_date <= c.timestamp <= end_date]
            
            if len(filtered_fallback) >= min_candles:
                # Sufficient data in fallback - use it!
                logger.info(f"✓ Using {len(filtered_fallback)} candles from fallback data (sufficient)")
                return filtered_fallback
            else:
                # Insufficient data in fallback - try Yahoo Finance first
                logger.warning(
                    f"⚠️ Fallback data has only {len(filtered_fallback)} candles "
                    f"(need {min_candles}). Trying Yahoo Finance..."
                )
                
                yahoo_candles = self._fetch_from_yahoo_finance(symbol, exchange, days)
                
                if yahoo_candles and len(yahoo_candles) >= min_candles:
                    logger.info(f"✓ Got {len(yahoo_candles)} candles from Yahoo Finance")
                    return yahoo_candles
                else:
                    # Yahoo failed or insufficient - use fallback anyway (with warning)
                    logger.warning(
                        f"⚠️ Yahoo Finance unavailable. Using {len(filtered_fallback)} "
                        f"fallback candles (may cause technical analysis errors)"
                    )
                    return filtered_fallback if filtered_fallback else []
        
        # No fallback data - try Yahoo Finance
        logger.info(f"No fallback data for {symbol}. Trying Yahoo Finance...")
        candles = self._fetch_from_yahoo_finance(symbol, exchange, days)
        
        if candles and len(candles) > 0:
            logger.info(f"✓ Got {len(candles)} candles from Yahoo Finance")
            return candles
        
        logger.error(f"❌ No data available for {symbol} from any source")
        return []
    
    def _fetch_from_java_service(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[CandleData]:
        """Fetch from Java microservice"""
        try:
            url = f"{self.java_service_url}/invest/historical"
            params = {
                "symbol": symbol,
                "exchange": exchange,
                "fromDate": start_date.strftime("%Y-%m-%d"),
                "toDate": end_date.strftime("%Y-%m-%d"),
                "interval": "day",
                "continuous": "true"
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_java_response(data)
            
            return []
            
        except Exception as e:
            logger.warning(f"Java service error: {str(e)}")
            return []
    
    def _fetch_from_yahoo_finance(
        self,
        symbol: str,
        exchange: str,
        days: int
    ) -> List[CandleData]:
        """
        Fetch from Yahoo Finance (FREE!)
        
        This is the magic - no API key needed!
        """
        try:
            import yfinance as yf
            
            # Convert symbol to Yahoo format
            if exchange == "NSE":
                yahoo_symbol = f"{symbol}.NS"
            elif exchange == "BSE":
                yahoo_symbol = f"{symbol}.BO"
            else:
                yahoo_symbol = symbol
            
            logger.info(f"Fetching {yahoo_symbol} from Yahoo Finance...")
            
            # Calculate period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data from Yahoo Finance for {yahoo_symbol}")
                return []
            
            # Convert to CandleData
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
            
            logger.info(f"✓ Yahoo Finance returned {len(candles)} candles for {symbol}")
            return candles
            
        except ImportError:
            logger.error("yfinance not installed! Run: pip install yfinance")
            return []
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {str(e)}")
            return []
    
    def _parse_java_response(self, data: dict) -> List[CandleData]:
        """Parse Java microservice response"""
        candles = []
        
        if 'data' in data and 'candles' in data['data']:
            for candle_data in data['data']['candles']:
                candle = CandleData(
                    timestamp=datetime.fromisoformat(candle_data[0]),
                    open=candle_data[1],
                    high=candle_data[2],
                    low=candle_data[3],
                    close=candle_data[4],
                    volume=candle_data[5]
                )
                candles.append(candle)
        
        return candles
    
    def candles_to_dataframe(self, candles: List[CandleData]) -> pd.DataFrame:
        """Convert candles to pandas DataFrame"""
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df


# ============================================================================
# 📝 EXAMPLE USAGE SCRIPT
# ============================================================================

def example_add_reliance():
    """
    Example: How to add RELIANCE stock from Screener.in JSON
    """
    
    # Step 1: Copy JSON from Screener.in
    RELIANCE_JSON = '''
    {
        "datasets": [
            {
                "metric": "Price",
                "label": "Price on NSE",
                "values": [
                    ["2025-01-02", "1305.25"],
                    ["2025-01-03", "1310.50"],
                    ["2025-01-06", "1298.75"],
                    ["2025-01-07", "1315.00"]
                ]
            }
        ]
    }
    '''
    
    # Step 2: Create DataService
    data_service = DataService()
    
    # Step 3: Add the stock
    success = data_service.add_stock_from_json('RELIANCE', RELIANCE_JSON)
    
    if success:
        print("✅ RELIANCE added successfully!")
        
        # Step 4: Test it
        candles = data_service.fetch_historical_data('RELIANCE', days=30)
        print(f"✅ Got {len(candles)} candles for RELIANCE")
    else:
        print("❌ Failed to add RELIANCE")


if __name__ == "__main__":
    example_add_reliance()

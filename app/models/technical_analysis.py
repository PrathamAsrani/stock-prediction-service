import pandas as pd
import numpy as np
from typing import List, Dict
import ta
from app.schemas.prediction_schema import CandleData, TechnicalIndicators

class TechnicalAnalyzer:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_indicators(candles: List[CandleData]) -> TechnicalIndicators:
        """Calculate all technical indicators"""
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in candles])
        
        df = df.sort_values('timestamp')
        
        # RSI
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
        
        # MACD
        macd_indicator = ta.trend.MACD(df['close'])
        macd = macd_indicator.macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        bollinger_upper = bollinger.bollinger_hband().iloc[-1]
        bollinger_lower = bollinger.bollinger_lband().iloc[-1]
        
        # Moving Averages
        sma_20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator().iloc[-1]
        sma_50 = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator().iloc[-1]
        ema_12 = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator().iloc[-1]
        ema_26 = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator().iloc[-1]
        
        # Volume SMA
        volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
        
        # ATR (Average True Range)
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[-1]
        
        return TechnicalIndicators(
            rsi=float(rsi),
            macd=float(macd),
            macd_signal=float(macd_signal),
            bollinger_upper=float(bollinger_upper),
            bollinger_lower=float(bollinger_lower),
            sma_20=float(sma_20),
            sma_50=float(sma_50),
            ema_12=float(ema_12),
            ema_26=float(ema_26),
            volume_sma=float(volume_sma),
            atr=float(atr)
        )
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for ML model"""
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Price position relative to BB
        df['price_to_bb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Trend strength
        df['trend_strength'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        return df
    
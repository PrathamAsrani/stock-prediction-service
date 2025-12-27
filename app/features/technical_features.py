import pandas as pd
import numpy as np
from typing import List, Dict
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Calculate comprehensive technical indicators and features"""
    
    @staticmethod
    def calculate_indicators(candles: List) -> 'TechnicalIndicators':
        """
        Calculate all technical indicators from candle data
        Returns TechnicalIndicators object
        """
        from app.schemas.prediction_schema import TechnicalIndicators, CandleData
        
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
        
        # Calculate indicators
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI
        rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
        
        # MACD
        macd_indicator = MACD(close)
        macd = macd_indicator.macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        
        # Bollinger Bands
        bollinger = BollingerBands(close, window=20, window_dev=2)
        bb_upper = bollinger.bollinger_hband().iloc[-1]
        bb_lower = bollinger.bollinger_lband().iloc[-1]
        
        # Moving Averages
        sma_20 = SMAIndicator(close, window=20).sma_indicator().iloc[-1]
        sma_50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]
        ema_12 = EMAIndicator(close, window=12).ema_indicator().iloc[-1]
        ema_26 = EMAIndicator(close, window=26).ema_indicator().iloc[-1]
        
        # Volume
        volume_sma = volume.rolling(window=20).mean().iloc[-1]
        
        # ATR
        atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        
        return TechnicalIndicators(
            rsi=float(rsi) if not np.isnan(rsi) else 50.0,
            macd=float(macd) if not np.isnan(macd) else 0.0,
            macd_signal=float(macd_signal) if not np.isnan(macd_signal) else 0.0,
            bollinger_upper=float(bb_upper) if not np.isnan(bb_upper) else close.iloc[-1],
            bollinger_lower=float(bb_lower) if not np.isnan(bb_lower) else close.iloc[-1],
            sma_20=float(sma_20) if not np.isnan(sma_20) else close.iloc[-1],
            sma_50=float(sma_50) if not np.isnan(sma_50) else close.iloc[-1],
            ema_12=float(ema_12) if not np.isnan(ema_12) else close.iloc[-1],
            ema_26=float(ema_26) if not np.isnan(ema_26) else close.iloc[-1],
            volume_sma=float(volume_sma) if not np.isnan(volume_sma) else 0.0,
            atr=float(atr) if not np.isnan(atr) else 1.0
        )
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical features for ML models
        """
        logger.info("Calculating technical features...")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # === PRICE-BASED FEATURES ===
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price changes
        df['price_change'] = df['close'] - df['close'].shift(1)
        df['high_low_range'] = df['high'] - df['low']
        df['open_close_range'] = abs(df['open'] - df['close'])
        
        # === VOLATILITY FEATURES ===
        
        # Historical volatility (20-day)
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_30'] = df['returns'].rolling(window=30).std()
        
        # ATR
        atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Bollinger Bands
        bollinger = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Keltner Channels
        keltner = KeltnerChannel(df['high'], df['low'], df['close'], window=20)
        df['kc_upper'] = keltner.keltner_channel_hband()
        df['kc_lower'] = keltner.keltner_channel_lband()
        
        # === MOMENTUM FEATURES ===
        
        # RSI
        rsi_indicator = RSIIndicator(df['close'], window=14)
        df['rsi'] = rsi_indicator.rsi()
        df['rsi_6'] = RSIIndicator(df['close'], window=6).rsi()
        df['rsi_21'] = RSIIndicator(df['close'], window=21).rsi()
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Rate of Change (ROC)
        df['roc'] = ROCIndicator(df['close'], window=10).roc()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # === TREND FEATURES ===
        
        # Moving Averages
        df['sma_5'] = SMAIndicator(df['close'], window=5).sma_indicator()
        df['sma_10'] = SMAIndicator(df['close'], window=10).sma_indicator()
        df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma_100'] = SMAIndicator(df['close'], window=100).sma_indicator()
        df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
        
        # Exponential Moving Averages
        df['ema_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
        df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
        
        # Moving Average Crosses
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ema_12_26_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
        
        # Price position relative to MAs
        df['close_to_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['close_to_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # ADX (Trend Strength)
        adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()
        
        # === VOLUME FEATURES ===
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # On-Balance Volume
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        
        # VWAP
        try:
            df['vwap'] = VolumeWeightedAveragePrice(
                df['high'], df['low'], df['close'], df['volume']
            ).volume_weighted_average_price()
        except:
            df['vwap'] = df['close']  # Fallback
        
        # Volume-price trend
        df['volume_price_trend'] = ((df['close'] - df['close'].shift(1)) / 
                                     df['close'].shift(1) * df['volume']).cumsum()
        
        # === STATISTICAL FEATURES ===
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'returns_mean_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
            df[f'high_max_{window}'] = df['high'].rolling(window=window).max()
            df[f'low_min_{window}'] = df['low'].rolling(window=window).min()
        
        # === LAG FEATURES ===
        
        # Price lags
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # === CANDLESTICK PATTERNS (Simplified) ===
        
        # Doji
        df['is_doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1).astype(int)
        
        # Hammer/Hanging Man
        body = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        df['is_hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)
        
        # Engulfing
        df['is_bullish_engulf'] = (
            (df['close'] > df['open']) & 
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)
        
        # === DERIVED FEATURES ===
        
        # Trend strength
        df['trend_strength'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # Volatility ratio
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
        
        # Volume surge
        df['volume_surge'] = (df['volume'] > df['volume_sma'] * 1.5).astype(int)
        
        logger.info(f"Technical features calculated. Total features: {len(df.columns)}")
        
        return df
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all technical feature names"""
        
        # This should match the features created in calculate_features
        base_features = [
            'returns', 'log_returns', 'price_change', 'high_low_range', 'open_close_range',
            'volatility', 'volatility_30', 'atr', 'atr_percent',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'kc_upper', 'kc_lower',
            'rsi', 'rsi_6', 'rsi_21',
            'macd', 'macd_signal', 'macd_diff',
            'stoch_k', 'stoch_d',
            'roc', 'momentum', 'momentum_5', 'momentum_20',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_9', 'ema_12', 'ema_26', 'ema_50',
            'sma_20_50_cross', 'ema_12_26_cross',
            'close_to_sma20', 'close_to_sma50',
            'adx', 'adx_pos', 'adx_neg',
            'volume_change', 'volume_sma', 'volume_ratio',
            'obv', 'obv_sma', 'vwap', 'volume_price_trend',
            'trend_strength', 'volatility_ratio', 'volume_surge',
            'is_doji', 'is_hammer', 'is_bullish_engulf'
        ]
        
        # Add rolling statistics features
        for window in [5, 10, 20]:
            base_features.extend([
                f'close_std_{window}',
                f'returns_mean_{window}',
                f'returns_std_{window}',
                f'high_max_{window}',
                f'low_min_{window}'
            ])
        
        # Add lag features
        for lag in [1, 2, 3, 5, 10, 20]:
            base_features.extend([
                f'close_lag_{lag}',
                f'volume_lag_{lag}',
                f'returns_lag_{lag}'
            ])
        
        return base_features
    
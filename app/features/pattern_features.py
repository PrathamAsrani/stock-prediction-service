import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class PatternFeaturesExtractor:
    """Extract chart pattern features"""
    
    @staticmethod
    def extract_features(df: pd.DataFrame) -> Dict:
        """
        Extract pattern-based features from price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of pattern features
        """
        features = {}
        
        # === CANDLESTICK PATTERNS ===
        candlestick_patterns = PatternFeaturesExtractor._detect_candlestick_patterns(df)
        features.update(candlestick_patterns)
        
        # === PRICE PATTERNS ===
        price_patterns = PatternFeaturesExtractor._detect_price_patterns(df)
        features.update(price_patterns)
        
        # === SUPPORT/RESISTANCE ===
        support_resistance = PatternFeaturesExtractor._calculate_support_resistance(df)
        features.update(support_resistance)
        
        # === BREAKOUT DETECTION ===
        breakout_features = PatternFeaturesExtractor._detect_breakouts(df)
        features.update(breakout_features)
        
        logger.info(f"Extracted {len(features)} pattern features")
        
        return features
    
    @staticmethod
    def _detect_candlestick_patterns(df: pd.DataFrame) -> Dict:
        """Detect candlestick patterns"""
        
        features = {}
        
        # Get last 5 candles
        recent = df.tail(5)
        
        if len(recent) < 2:
            return features
        
        # Calculate body and shadows
        body = abs(recent['close'] - recent['open'])
        total_range = recent['high'] - recent['low']
        upper_shadow = recent['high'] - recent[['open', 'close']].max(axis=1)
        lower_shadow = recent[['open', 'close']].min(axis=1) - recent['low']
        
        # === SINGLE CANDLE PATTERNS ===
        
        # Doji (last candle)
        last_body = body.iloc[-1]
        last_range = total_range.iloc[-1]
        features['is_doji'] = 1 if last_body < (last_range * 0.1) else 0
        
        # Hammer
        last_lower_shadow = lower_shadow.iloc[-1]
        last_upper_shadow = upper_shadow.iloc[-1]
        features['is_hammer'] = 1 if (
            last_lower_shadow > (2 * last_body) and 
            last_upper_shadow < last_body
        ) else 0
        
        # Shooting Star
        features['is_shooting_star'] = 1 if (
            last_upper_shadow > (2 * last_body) and 
            last_lower_shadow < last_body
        ) else 0
        
        # === TWO CANDLE PATTERNS ===
        
        if len(recent) >= 2:
            # Engulfing (bullish/bearish)
            prev = recent.iloc[-2]
            curr = recent.iloc[-1]
            
            features['is_bullish_engulfing'] = 1 if (
                curr['close'] > curr['open'] and
                prev['close'] < prev['open'] and
                curr['open'] < prev['close'] and
                curr['close'] > prev['open']
            ) else 0
            
            features['is_bearish_engulfing'] = 1 if (
                curr['close'] < curr['open'] and
                prev['close'] > prev['open'] and
                curr['open'] > prev['close'] and
                curr['close'] < prev['open']
            ) else 0
        
        return features
    
    @staticmethod
    def _detect_price_patterns(df: pd.DataFrame) -> Dict:
        """Detect price chart patterns"""
        
        features = {}
        
        if len(df) < 20:
            return features
        
        recent = df.tail(60)  # Last 60 days
        close = recent['close'].values
        
        # === DOUBLE TOP/BOTTOM ===
        peaks = PatternFeaturesExtractor._find_peaks(close, distance=10)
        troughs = PatternFeaturesExtractor._find_troughs(close, distance=10)
        
        features['num_recent_peaks'] = len(peaks)
        features['num_recent_troughs'] = len(troughs)
        
        # Check for double top
        if len(peaks) >= 2:
            last_two_peaks = close[peaks[-2:]]
            features['potential_double_top'] = 1 if abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0] < 0.03 else 0
        else:
            features['potential_double_top'] = 0
        
        # Check for double bottom
        if len(troughs) >= 2:
            last_two_troughs = close[troughs[-2:]]
            features['potential_double_bottom'] = 1 if abs(last_two_troughs[0] - last_two_troughs[1]) / last_two_troughs[0] < 0.03 else 0
        else:
            features['potential_double_bottom'] = 0
        
        # === TREND PATTERNS ===
        
        # Higher highs, higher lows (uptrend)
        if len(peaks) >= 2 and len(troughs) >= 2:
            features['higher_highs'] = 1 if all(close[peaks[i]] < close[peaks[i+1]] for i in range(len(peaks)-1)) else 0
            features['higher_lows'] = 1 if all(close[troughs[i]] < close[troughs[i+1]] for i in range(len(troughs)-1)) else 0
            features['in_uptrend'] = features['higher_highs'] and features['higher_lows']
            
            # Lower highs, lower lows (downtrend)
            features['lower_highs'] = 1 if all(close[peaks[i]] > close[peaks[i+1]] for i in range(len(peaks)-1)) else 0
            features['lower_lows'] = 1 if all(close[troughs[i]] > close[troughs[i+1]] for i in range(len(troughs)-1)) else 0
            features['in_downtrend'] = features['lower_highs'] and features['lower_lows']
        
        return features
    
    @staticmethod
    def _calculate_support_resistance(df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        
        features = {}
        
        if len(df) < 20:
            return features
        
        recent = df.tail(60)
        close = recent['close']
        current_price = close.iloc[-1]
        
        # Find local maxima and minima
        highs = recent['high'].values
        lows = recent['low'].values
        
        peaks = PatternFeaturesExtractor._find_peaks(highs, distance=5)
        troughs = PatternFeaturesExtractor._find_troughs(lows, distance=5)
        
        # Resistance levels (peaks)
        if len(peaks) > 0:
            resistance_levels = highs[peaks]
            nearest_resistance = resistance_levels[resistance_levels > current_price].min() if any(resistance_levels > current_price) else None
            
            if nearest_resistance:
                features['distance_to_resistance'] = (nearest_resistance - current_price) / current_price
            else:
                features['distance_to_resistance'] = 0.1  # Far from resistance
        
        # Support levels (troughs)
        if len(troughs) > 0:
            support_levels = lows[troughs]
            nearest_support = support_levels[support_levels < current_price].max() if any(support_levels < current_price) else None
            
            if nearest_support:
                features['distance_to_support'] = (current_price - nearest_support) / current_price
            else:
                features['distance_to_support'] = 0.1  # Far from support
        
        # Position within range
        if 'distance_to_resistance' in features and 'distance_to_support' in features:
            total_range = features['distance_to_resistance'] + features['distance_to_support']
            if total_range > 0:
                features['position_in_range'] = features['distance_to_support'] / total_range
        
        return features
    
    @staticmethod
    def _detect_breakouts(df: pd.DataFrame) -> Dict:
        """Detect breakout patterns"""
        
        features = {}
        
        if len(df) < 20:
            return features
        
        recent = df.tail(20)
        current_price = recent['close'].iloc[-1]
        
        # 20-day high/low
        high_20 = recent['high'].max()
        low_20 = recent['low'].min()
        
        # Breakout detection
        features['near_20d_high'] = 1 if current_price >= high_20 * 0.98 else 0
        features['near_20d_low'] = 1 if current_price <= low_20 * 1.02 else 0
        
        # Consolidation detection (low volatility)
        price_range = (high_20 - low_20) / current_price
        features['is_consolidating'] = 1 if price_range < 0.05 else 0
        
        return features
    
    @staticmethod
    def _find_peaks(data: np.ndarray, distance: int = 5) -> np.ndarray:
        """Find local peaks in data"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(data, distance=distance)
        return peaks
    
    @staticmethod
    def _find_troughs(data: np.ndarray, distance: int = 5) -> np.ndarray:
        """Find local troughs in data"""
        from scipy.signal import find_peaks
        troughs, _ = find_peaks(-data, distance=distance)
        return troughs
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of pattern feature names"""
        return [
            # Candlestick patterns
            'is_doji', 'is_hammer', 'is_shooting_star',
            'is_bullish_engulfing', 'is_bearish_engulfing',
            
            # Price patterns
            'num_recent_peaks', 'num_recent_troughs',
            'potential_double_top', 'potential_double_bottom',
            'higher_highs', 'higher_lows', 'in_uptrend',
            'lower_highs', 'lower_lows', 'in_downtrend',
            
            # Support/Resistance
            'distance_to_resistance', 'distance_to_support', 'position_in_range',
            
            # Breakouts
            'near_20d_high', 'near_20d_low', 'is_consolidating'
        ]
    
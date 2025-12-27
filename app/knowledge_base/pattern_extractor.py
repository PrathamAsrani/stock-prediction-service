import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class PatternExtractor:
    """Extract and document chart patterns from historical data"""
    
    def __init__(self):
        self.patterns = {
            'head_and_shoulders': self._detect_head_and_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'cup_and_handle': self._detect_cup_and_handle,
            'ascending_triangle': self._detect_ascending_triangle,
            'descending_triangle': self._detect_descending_triangle,
            'bullish_flag': self._detect_bullish_flag,
            'bearish_flag': self._detect_bearish_flag,
        }
    
    def extract_patterns_from_history(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> List[Document]:
        """Extract all patterns from historical data and create knowledge base entries"""
        
        documents = []
        
        for pattern_name, detector_func in self.patterns.items():
            detected_patterns = detector_func(df)
            
            for pattern_info in detected_patterns:
                # Create descriptive text about the pattern
                pattern_text = self._generate_pattern_description(
                    pattern_name,
                    pattern_info,
                    symbol
                )
                
                doc = Document(
                    page_content=pattern_text,
                    metadata={
                        "type": "pattern",
                        "symbol": symbol,
                        "pattern_name": pattern_name,
                        "start_date": str(pattern_info['start_date']),
                        "end_date": str(pattern_info['end_date']),
                        "outcome": pattern_info.get('outcome'),
                        "price_change": pattern_info.get('price_change'),
                        **pattern_info.get('metrics', {})
                    }
                )
                documents.append(doc)
        
        logger.info(f"Extracted {len(documents)} patterns for {symbol}")
        
        return documents
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """Detect head and shoulders pattern"""
        patterns = []
        window = 60  # Look for pattern in 60-day windows
        
        for i in range(window, len(df) - window):
            segment = df.iloc[i-window:i+window]
            
            # Find peaks
            peaks = self._find_peaks(segment['high'].values)
            
            if len(peaks) >= 3:
                # Check if middle peak is highest (head)
                middle_idx = len(peaks) // 2
                if peaks[middle_idx] > peaks[middle_idx-1] and peaks[middle_idx] > peaks[middle_idx+1]:
                    # Verify shoulders are roughly equal
                    left_shoulder = peaks[middle_idx-1]
                    right_shoulder = peaks[middle_idx+1]
                    
                    if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:  # Within 5%
                        # Calculate outcome
                        pattern_end = i + window
                        if pattern_end + 30 < len(df):
                            future_price = df.iloc[pattern_end + 30]['close']
                            current_price = df.iloc[pattern_end]['close']
                            price_change = ((future_price - current_price) / current_price) * 100
                            outcome = 'bullish' if price_change > 0 else 'bearish'
                        else:
                            price_change = None
                            outcome = 'unknown'
                        
                        patterns.append({
                            'start_date': df.index[i-window],
                            'end_date': df.index[i+window],
                            'outcome': outcome,
                            'price_change': price_change,
                            'metrics': {
                                'head_height': peaks[middle_idx],
                                'shoulder_avg': (left_shoulder + right_shoulder) / 2
                            }
                        })
        
        return patterns
    
    def _detect_double_top(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double top pattern (bearish)"""
        patterns = []
        window = 40
        
        for i in range(window, len(df) - window):
            segment = df.iloc[i-window:i+window]
            peaks = self._find_peaks(segment['high'].values)
            
            if len(peaks) >= 2:
                # Check if two peaks are roughly equal
                if abs(peaks[-1] - peaks[-2]) / peaks[-1] < 0.03:  # Within 3%
                    pattern_end = i + window
                    if pattern_end + 20 < len(df):
                        future_price = df.iloc[pattern_end + 20]['close']
                        current_price = df.iloc[pattern_end]['close']
                        price_change = ((future_price - current_price) / current_price) * 100
                        
                        patterns.append({
                            'start_date': df.index[i-window],
                            'end_date': df.index[i+window],
                            'outcome': 'bearish',
                            'price_change': price_change,
                            'metrics': {
                                'peak_level': (peaks[-1] + peaks[-2]) / 2
                            }
                        })
        
        return patterns
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double bottom pattern (bullish)"""
        patterns = []
        window = 40
        
        for i in range(window, len(df) - window):
            segment = df.iloc[i-window:i+window]
            troughs = self._find_troughs(segment['low'].values)
            
            if len(troughs) >= 2:
                if abs(troughs[-1] - troughs[-2]) / troughs[-1] < 0.03:
                    pattern_end = i + window
                    if pattern_end + 20 < len(df):
                        future_price = df.iloc[pattern_end + 20]['close']
                        current_price = df.iloc[pattern_end]['close']
                        price_change = ((future_price - current_price) / current_price) * 100
                        
                        patterns.append({
                            'start_date': df.index[i-window],
                            'end_date': df.index[i+window],
                            'outcome': 'bullish',
                            'price_change': price_change,
                            'metrics': {
                                'support_level': (troughs[-1] + troughs[-2]) / 2
                            }
                        })
        
        return patterns
    
    def _detect_cup_and_handle(self, df: pd.DataFrame) -> List[Dict]:
        """Detect cup and handle pattern (bullish)"""
        patterns = []
        # Implementation similar to above
        return patterns
    
    def _detect_ascending_triangle(self, df: pd.DataFrame) -> List[Dict]:
        """Detect ascending triangle (bullish)"""
        patterns = []
        # Implementation for ascending triangle
        return patterns
    
    def _detect_descending_triangle(self, df: pd.DataFrame) -> List[Dict]:
        """Detect descending triangle (bearish)"""
        patterns = []
        # Implementation for descending triangle
        return patterns
    
    def _detect_bullish_flag(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bullish flag pattern"""
        patterns = []
        # Implementation for bullish flag
        return patterns
    
    def _detect_bearish_flag(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bearish flag pattern"""
        patterns = []
        # Implementation for bearish flag
        return patterns
    
    def _find_peaks(self, data: np.ndarray, threshold: float = 0.02) -> List[float]:
        """Find local peaks in data"""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                if not peaks or (data[i] - peaks[-1]) / peaks[-1] > threshold:
                    peaks.append(data[i])
        return peaks
    
    def _find_troughs(self, data: np.ndarray, threshold: float = 0.02) -> List[float]:
        """Find local troughs in data"""
        troughs = []
        for i in range(1, len(data) - 1):
            if data[i] < data[i-1] and data[i] < data[i+1]:
                if not troughs or (troughs[-1] - data[i]) / troughs[-1] > threshold:
                    troughs.append(data[i])
        return troughs
    
    def _generate_pattern_description(
        self, 
        pattern_name: str, 
        pattern_info: Dict,
        symbol: str
    ) -> str:
        """Generate natural language description of pattern"""
        
        pattern_descriptions = {
            'head_and_shoulders': f"A head and shoulders pattern was observed in {symbol} from {pattern_info['start_date']} to {pattern_info['end_date']}. This bearish reversal pattern typically indicates an upcoming downtrend. The pattern resulted in a {pattern_info.get('outcome', 'unknown')} outcome with a price change of {pattern_info.get('price_change', 'N/A')}%.",
            
            'double_top': f"A double top pattern formed in {symbol} between {pattern_info['start_date']} and {pattern_info['end_date']}. This bearish pattern suggests resistance at the peak level of {pattern_info.get('metrics', {}).get('peak_level', 'N/A')}. The subsequent price movement was {pattern_info.get('outcome', 'unknown')} with a {pattern_info.get('price_change', 'N/A')}% change.",
            
            'double_bottom': f"A double bottom pattern was detected in {symbol} from {pattern_info['start_date']} to {pattern_info['end_date']}. This bullish reversal pattern indicated strong support at {pattern_info.get('metrics', {}).get('support_level', 'N/A')}. The pattern led to a {pattern_info.get('outcome', 'unknown')} move with {pattern_info.get('price_change', 'N/A')}% price change.",
        }
        
        description = pattern_descriptions.get(
            pattern_name,
            f"A {pattern_name.replace('_', ' ')} pattern was identified in {symbol} from {pattern_info['start_date']} to {pattern_info['end_date']}. Outcome: {pattern_info.get('outcome', 'unknown')}, Price change: {pattern_info.get('price_change', 'N/A')}%."
        )
        
        return description
    
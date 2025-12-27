import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class FundamentalPreprocessor:
    """Preprocess fundamental data"""
    
    @staticmethod
    def clean_fundamentals(data: Dict) -> Dict:
        """Clean fundamental metrics"""
        
        cleaned = {}
        
        for key, value in data.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                cleaned[key] = None
            elif isinstance(value, (int, float)):
                # Remove extreme outliers
                if abs(value) > 1e10:
                    cleaned[key] = None
                else:
                    cleaned[key] = float(value)
            else:
                cleaned[key] = value
        
        return cleaned
    
    @staticmethod
    def calculate_derived_metrics(data: Dict) -> Dict:
        """Calculate additional fundamental metrics"""
        
        # P/E to Growth (PEG) ratio
        if data.get('pe_ratio') and data.get('earnings_growth'):
            if data['earnings_growth'] != 0:
                data['peg_ratio'] = data['pe_ratio'] / (data['earnings_growth'] * 100)
        
        # Return on Invested Capital (ROIC)
        # Simplified calculation
        if data.get('roa') and data.get('debt_to_equity'):
            data['roic_estimate'] = data['roa'] * (1 + data['debt_to_equity'] / 100)
        
        # Altman Z-Score (simplified)
        # Requires more detailed financial data
        
        return data
    
    @staticmethod
    def normalize_ratios(data: Dict) -> Dict:
        """Normalize financial ratios to 0-1 scale"""
        
        normalized = data.copy()
        
        # P/E ratio (normalize to 0-50 range)
        if data.get('pe_ratio'):
            normalized['pe_ratio_norm'] = min(data['pe_ratio'] / 50, 1.0)
        
        # ROE (normalize to 0-30% range)
        if data.get('roe'):
            normalized['roe_norm'] = min(data['roe'] / 0.3, 1.0)
        
        # Debt to Equity (normalize to 0-200 range)
        if data.get('debt_to_equity'):
            normalized['de_ratio_norm'] = min(data['debt_to_equity'] / 200, 1.0)
        
        return normalized
    
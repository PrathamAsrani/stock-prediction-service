import yfinance as yf
import pandas as pd
from typing import Dict, Optional
import logging
from app.schemas.prediction_schema import FundamentalData

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """Analyze fundamental metrics"""
    
    @staticmethod
    def fetch_fundamentals(symbol: str, exchange: str = "NSE") -> Optional[FundamentalData]:
        """Fetch fundamental data from Yahoo Finance"""
        try:
            # Convert to Yahoo Finance symbol format
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
            
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            fundamental_data = FundamentalData(
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                roe=info.get('returnOnEquity'),
                debt_to_equity=info.get('debtToEquity'),
                dividend_yield=info.get('dividendYield'),
                market_cap=info.get('marketCap'),
                revenue_growth=info.get('revenueGrowth'),
                profit_margin=info.get('profitMargins')
            )
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
            return None
    
    @staticmethod
    def calculate_fundamental_score(data: FundamentalData) -> float:
        """
        Calculate fundamental health score (0-1)
        Higher score = better fundamentals
        """
        score = 0.0
        weights = {
            'pe_ratio': 0.15,
            'pb_ratio': 0.10,
            'roe': 0.20,
            'debt_to_equity': 0.15,
            'dividend_yield': 0.10,
            'revenue_growth': 0.15,
            'profit_margin': 0.15
        }
        
        # PE Ratio (lower is better, ideal < 25)
        if data.pe_ratio is not None:
            if data.pe_ratio < 15:
                score += weights['pe_ratio'] * 1.0
            elif data.pe_ratio < 25:
                score += weights['pe_ratio'] * 0.7
            elif data.pe_ratio < 35:
                score += weights['pe_ratio'] * 0.4
            else:
                score += weights['pe_ratio'] * 0.2
        
        # PB Ratio (lower is better, ideal < 3)
        if data.pb_ratio is not None:
            if data.pb_ratio < 1.5:
                score += weights['pb_ratio'] * 1.0
            elif data.pb_ratio < 3:
                score += weights['pb_ratio'] * 0.7
            elif data.pb_ratio < 5:
                score += weights['pb_ratio'] * 0.4
            else:
                score += weights['pb_ratio'] * 0.2
        
        # ROE (higher is better, ideal > 15%)
        if data.roe is not None:
            roe_percent = data.roe * 100
            if roe_percent > 20:
                score += weights['roe'] * 1.0
            elif roe_percent > 15:
                score += weights['roe'] * 0.8
            elif roe_percent > 10:
                score += weights['roe'] * 0.5
            else:
                score += weights['roe'] * 0.2
        
        # Debt to Equity (lower is better, ideal < 1)
        if data.debt_to_equity is not None:
            if data.debt_to_equity < 0.5:
                score += weights['debt_to_equity'] * 1.0
            elif data.debt_to_equity < 1:
                score += weights['debt_to_equity'] * 0.8
            elif data.debt_to_equity < 2:
                score += weights['debt_to_equity'] * 0.5
            else:
                score += weights['debt_to_equity'] * 0.2
        
        # Dividend Yield (higher is better)
        if data.dividend_yield is not None:
            dividend_percent = data.dividend_yield * 100
            if dividend_percent > 3:
                score += weights['dividend_yield'] * 1.0
            elif dividend_percent > 2:
                score += weights['dividend_yield'] * 0.7
            elif dividend_percent > 1:
                score += weights['dividend_yield'] * 0.5
            else:
                score += weights['dividend_yield'] * 0.3
        
        # Revenue Growth (higher is better)
        if data.revenue_growth is not None:
            growth_percent = data.revenue_growth * 100
            if growth_percent > 20:
                score += weights['revenue_growth'] * 1.0
            elif growth_percent > 10:
                score += weights['revenue_growth'] * 0.8
            elif growth_percent > 5:
                score += weights['revenue_growth'] * 0.6
            else:
                score += weights['revenue_growth'] * 0.3
        
        # Profit Margin (higher is better)
        if data.profit_margin is not None:
            margin_percent = data.profit_margin * 100
            if margin_percent > 15:
                score += weights['profit_margin'] * 1.0
            elif margin_percent > 10:
                score += weights['profit_margin'] * 0.8
            elif margin_percent > 5:
                score += weights['profit_margin'] * 0.6
            else:
                score += weights['profit_margin'] * 0.3
        
        # Normalize to 0-1 range
        max_possible_score = sum(weights.values())
        normalized_score = score / max_possible_score
        
        return min(1.0, max(0.0, normalized_score))
    
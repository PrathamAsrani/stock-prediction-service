import yfinance as yf
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FundamentalsCollector:
    """Collect fundamental data for stocks"""
    
    def fetch_fundamentals(self, symbol: str, exchange: str = "NSE") -> Dict:
        """Fetch fundamental metrics"""
        try:
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            fundamentals = {
                'symbol': symbol,
                'exchange': exchange,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'beta': info.get('beta'),
                '52w_high': info.get('fiftyTwoWeekHigh'),
                '52w_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
            }
            
            logger.info(f"Fetched fundamentals for {symbol}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
            return {}
    
    def fetch_financials(self, symbol: str, exchange: str = "NSE") -> Dict:
        """Fetch financial statements"""
        try:
            yahoo_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
            ticker = yf.Ticker(yahoo_symbol)
            
            financials = {
                'income_statement': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow,
                'quarterly_financials': ticker.quarterly_financials,
                'quarterly_balance_sheet': ticker.quarterly_balance_sheet,
                'quarterly_cashflow': ticker.quarterly_cashflow
            }
            
            return financials
            
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {str(e)}")
            return {}
        
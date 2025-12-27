import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FundamentalFeaturesExtractor:
    """Extract and engineer fundamental analysis features"""
    
    @staticmethod
    def extract_features(fundamentals: Dict) -> Dict:
        """
        Extract fundamental features for ML model
        
        Args:
            fundamentals: Dictionary of fundamental metrics
            
        Returns:
            Dictionary of engineered features
        """
        features = {}
        
        # === VALUATION RATIOS ===
        
        # P/E Ratio features
        pe = fundamentals.get('pe_ratio')
        if pe:
            features['pe_ratio'] = pe
            features['pe_ratio_log'] = np.log1p(pe) if pe > 0 else 0
            features['pe_category'] = FundamentalFeaturesExtractor._categorize_pe(pe)
        
        # P/B Ratio
        pb = fundamentals.get('pb_ratio')
        if pb:
            features['pb_ratio'] = pb
            features['pb_ratio_log'] = np.log1p(pb) if pb > 0 else 0
        
        # P/S Ratio
        ps = fundamentals.get('ps_ratio')
        if ps:
            features['ps_ratio'] = ps
        
        # === PROFITABILITY METRICS ===
        
        # ROE
        roe = fundamentals.get('roe')
        if roe:
            features['roe'] = roe * 100  # Convert to percentage
            features['roe_category'] = FundamentalFeaturesExtractor._categorize_roe(roe)
        
        # ROA
        roa = fundamentals.get('roa')
        if roa:
            features['roa'] = roa * 100
        
        # Profit Margins
        profit_margin = fundamentals.get('profit_margin')
        if profit_margin:
            features['profit_margin'] = profit_margin * 100
        
        operating_margin = fundamentals.get('operating_margin')
        if operating_margin:
            features['operating_margin'] = operating_margin * 100
        
        gross_margin = fundamentals.get('gross_margin')
        if gross_margin:
            features['gross_margin'] = gross_margin * 100
        
        # === FINANCIAL HEALTH ===
        
        # Debt to Equity
        de_ratio = fundamentals.get('debt_to_equity')
        if de_ratio:
            features['debt_to_equity'] = de_ratio
            features['de_ratio_category'] = FundamentalFeaturesExtractor._categorize_debt(de_ratio)
        
        # Current Ratio
        current_ratio = fundamentals.get('current_ratio')
        if current_ratio:
            features['current_ratio'] = current_ratio
            features['liquidity_healthy'] = 1 if current_ratio > 1.5 else 0
        
        # Quick Ratio
        quick_ratio = fundamentals.get('quick_ratio')
        if quick_ratio:
            features['quick_ratio'] = quick_ratio
        
        # === GROWTH METRICS ===
        
        # Revenue Growth
        revenue_growth = fundamentals.get('revenue_growth')
        if revenue_growth:
            features['revenue_growth'] = revenue_growth * 100
            features['revenue_growth_positive'] = 1 if revenue_growth > 0 else 0
        
        # Earnings Growth
        earnings_growth = fundamentals.get('earnings_growth')
        if earnings_growth:
            features['earnings_growth'] = earnings_growth * 100
            features['earnings_growth_positive'] = 1 if earnings_growth > 0 else 0
        
        # === DIVIDEND METRICS ===
        
        # Dividend Yield
        div_yield = fundamentals.get('dividend_yield')
        if div_yield:
            features['dividend_yield'] = div_yield * 100
            features['pays_dividend'] = 1 if div_yield > 0 else 0
        
        # Payout Ratio
        payout_ratio = fundamentals.get('payout_ratio')
        if payout_ratio:
            features['payout_ratio'] = payout_ratio * 100
        
        # === MARKET METRICS ===
        
        # Market Cap (in billions)
        market_cap = fundamentals.get('market_cap')
        if market_cap:
            features['market_cap_billions'] = market_cap / 1e9
            features['market_cap_category'] = FundamentalFeaturesExtractor._categorize_market_cap(market_cap)
        
        # Beta
        beta = fundamentals.get('beta')
        if beta:
            features['beta'] = beta
            features['volatility_level'] = FundamentalFeaturesExtractor._categorize_beta(beta)
        
        # === COMPOSITE SCORES ===
        
        # Quality Score (0-100)
        features['quality_score'] = FundamentalFeaturesExtractor._calculate_quality_score(fundamentals)
        
        # Value Score (0-100)
        features['value_score'] = FundamentalFeaturesExtractor._calculate_value_score(fundamentals)
        
        # Growth Score (0-100)
        features['growth_score'] = FundamentalFeaturesExtractor._calculate_growth_score(fundamentals)
        
        # Overall Fundamental Score
        features['fundamental_score'] = (
            features['quality_score'] * 0.4 +
            features['value_score'] * 0.3 +
            features['growth_score'] * 0.3
        )
        
        logger.info(f"Extracted {len(features)} fundamental features")
        
        return features
    
    @staticmethod
    def _categorize_pe(pe: float) -> int:
        """Categorize P/E ratio"""
        if pe < 0:
            return 0  # Negative earnings
        elif pe < 15:
            return 1  # Undervalued
        elif pe < 25:
            return 2  # Fair value
        elif pe < 35:
            return 3  # Slightly overvalued
        else:
            return 4  # Overvalued
    
    @staticmethod
    def _categorize_roe(roe: float) -> int:
        """Categorize ROE"""
        roe_pct = roe * 100
        if roe_pct < 5:
            return 0  # Poor
        elif roe_pct < 10:
            return 1  # Below average
        elif roe_pct < 15:
            return 2  # Average
        elif roe_pct < 20:
            return 3  # Good
        else:
            return 4  # Excellent
    
    @staticmethod
    def _categorize_debt(de_ratio: float) -> int:
        """Categorize Debt/Equity ratio"""
        if de_ratio < 0.3:
            return 0  # Very low debt
        elif de_ratio < 0.7:
            return 1  # Low debt
        elif de_ratio < 1.5:
            return 2  # Moderate debt
        elif de_ratio < 2.5:
            return 3  # High debt
        else:
            return 4  # Very high debt
    
    @staticmethod
    def _categorize_market_cap(market_cap: float) -> int:
        """Categorize market cap"""
        cap_billions = market_cap / 1e9
        if cap_billions < 5:
            return 0  # Small cap
        elif cap_billions < 20:
            return 1  # Mid cap
        else:
            return 2  # Large cap
    
    @staticmethod
    def _categorize_beta(beta: float) -> int:
        """Categorize beta (volatility)"""
        if beta < 0.8:
            return 0  # Low volatility
        elif beta < 1.2:
            return 1  # Market volatility
        else:
            return 2  # High volatility
    
    @staticmethod
    def _calculate_quality_score(fundamentals: Dict) -> float:
        """Calculate quality score (0-100)"""
        score = 0
        weights_sum = 0
        
        # ROE (weight: 30)
        roe = fundamentals.get('roe')
        if roe:
            roe_pct = roe * 100
            roe_score = min(100, (roe_pct / 20) * 100)
            score += roe_score * 0.30
            weights_sum += 0.30
        
        # Profit Margin (weight: 25)
        profit_margin = fundamentals.get('profit_margin')
        if profit_margin:
            pm_pct = profit_margin * 100
            pm_score = min(100, (pm_pct / 15) * 100)
            score += pm_score * 0.25
            weights_sum += 0.25
        
        # Debt/Equity (weight: 25) - lower is better
        de_ratio = fundamentals.get('debt_to_equity')
        if de_ratio:
            de_score = max(0, 100 - (de_ratio / 2) * 100)
            score += de_score * 0.25
            weights_sum += 0.25
        
        # Current Ratio (weight: 20)
        current_ratio = fundamentals.get('current_ratio')
        if current_ratio:
            cr_score = min(100, (current_ratio / 2) * 100)
            score += cr_score * 0.20
            weights_sum += 0.20
        
        # Normalize
        if weights_sum > 0:
            return score / weights_sum
        return 50.0  # Default neutral score
    
    @staticmethod
    def _calculate_value_score(fundamentals: Dict) -> float:
        """Calculate value score (0-100)"""
        score = 0
        weights_sum = 0
        
        # P/E Ratio (weight: 40) - lower is better
        pe = fundamentals.get('pe_ratio')
        if pe and pe > 0:
            pe_score = max(0, 100 - (pe / 30) * 100)
            score += pe_score * 0.40
            weights_sum += 0.40
        
        # P/B Ratio (weight: 30) - lower is better
        pb = fundamentals.get('pb_ratio')
        if pb and pb > 0:
            pb_score = max(0, 100 - (pb / 3) * 100)
            score += pb_score * 0.30
            weights_sum += 0.30
        
        # Dividend Yield (weight: 30) - higher is better
        div_yield = fundamentals.get('dividend_yield')
        if div_yield:
            div_pct = div_yield * 100
            div_score = min(100, (div_pct / 4) * 100)
            score += div_score * 0.30
            weights_sum += 0.30
        
        if weights_sum > 0:
            return score / weights_sum
        return 50.0
    
    @staticmethod
    def _calculate_growth_score(fundamentals: Dict) -> float:
        """Calculate growth score (0-100)"""
        score = 0
        weights_sum = 0
        
        # Revenue Growth (weight: 50)
        revenue_growth = fundamentals.get('revenue_growth')
        if revenue_growth:
            rg_pct = revenue_growth * 100
            rg_score = min(100, max(0, 50 + rg_pct * 2))
            score += rg_score * 0.50
            weights_sum += 0.50
        
        # Earnings Growth (weight: 50)
        earnings_growth = fundamentals.get('earnings_growth')
        if earnings_growth:
            eg_pct = earnings_growth * 100
            eg_score = min(100, max(0, 50 + eg_pct * 2))
            score += eg_score * 0.50
            weights_sum += 0.50
        
        if weights_sum > 0:
            return score / weights_sum
        return 50.0
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of fundamental feature names"""
        return [
            'pe_ratio', 'pe_ratio_log', 'pe_category',
            'pb_ratio', 'pb_ratio_log',
            'ps_ratio',
            'roe', 'roe_category',
            'roa',
            'profit_margin', 'operating_margin', 'gross_margin',
            'debt_to_equity', 'de_ratio_category',
            'current_ratio', 'liquidity_healthy',
            'quick_ratio',
            'revenue_growth', 'revenue_growth_positive',
            'earnings_growth', 'earnings_growth_positive',
            'dividend_yield', 'pays_dividend',
            'payout_ratio',
            'market_cap_billions', 'market_cap_category',
            'beta', 'volatility_level',
            'quality_score', 'value_score', 'growth_score', 'fundamental_score'
        ]
    
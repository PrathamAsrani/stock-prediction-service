import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

from app.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    TechnicalIndicators,
    CandleData
)
from app.models.ensemble_model import EnsembleModel
from app.features.technical_features import TechnicalAnalyzer
from app.services.data_service import DataService
from app.services.rag_service import RAGService
from app.config.settings import settings

logger = logging.getLogger(__name__)

class StockPredictionService:
    """Enhanced prediction service with ensemble model and RAG"""
    
    def __init__(self):
        self.ensemble_model = EnsembleModel(
            use_xgboost=settings.ENABLE_XGBOOST,
            use_lstm=settings.ENABLE_LSTM,
            use_transformer=settings.ENABLE_TRANSFORMER
        )
        self.data_service = DataService()
        self.rag_service = RAGService()
        
        # Load trained model
        try:
            self.ensemble_model.load()
            logger.info("✓ Ensemble model loaded successfully")
        except FileNotFoundError:
            logger.warning("⚠ No trained model found. Please train the model first.")
    
    async def predict_stock_growth(self, request: PredictionRequest) -> PredictionResponse:
        """
        Main prediction function with RAG-enhanced analysis
        
        Analyzes stock using:
        1. Ensemble ML predictions (XGBoost + LSTM + Transformer)
        2. Technical analysis
        3. Knowledge base insights (books, reports, patterns)
        4. Historical pattern matching
        """
        logger.info(f"Starting enhanced prediction for {request.symbol}")
        
        try:
            # Step 1: Fetch or use provided historical data
            candles = await self._get_historical_data(request)
            
            # Step 2: Calculate technical indicators
            technical_indicators = TechnicalAnalyzer.calculate_indicators(candles)
            
            # Step 3: Prepare features for ML model
            df = self.data_service.candles_to_dataframe(candles)
            df = TechnicalAnalyzer.calculate_features(df)
            
            # Step 4: Get RAG insights from knowledge base
            logger.info("Retrieving insights from knowledge base...")
            rag_insights = self.rag_service.get_trading_insights(
                symbol=request.symbol,
                query="swing trading strategy analysis",
                include_patterns=True,
                include_fundamentals=True
            )
            
            # Step 5: Get swing trading strategy
            trading_strategy = self.rag_service.get_swing_trading_strategy(request.symbol)
            
            # ==================== CRITICAL FIX HERE ====================
            # Step 6: Make ensemble prediction
            logger.info("Making ensemble prediction...")
            
            # Prepare features - exclude metadata columns only
            # The technical features are already calculated by TechnicalAnalyzer
            exclude_cols = ['symbol']
            if 'target' in df.columns:
                exclude_cols.append('target')
            
            # Select only numeric feature columns (all the technical indicators)
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            features = df[feature_cols].fillna(0)  # Fill any remaining NaN with 0
            features_latest = features.tail(1)
            # ===========================================================
            
            # Get detailed prediction breakdown
            prediction_breakdown = self.ensemble_model.predict_with_breakdown(
                features_latest,
                symbols=[request.symbol]
            )
            
            # Extract ensemble prediction
            ensemble_pred = prediction_breakdown['ensemble_prediction']
            ensemble_prob = prediction_breakdown['ensemble_probability']
            
            # Step 7: Calculate metrics
            current_price = df['close'].iloc[-1]
            volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.02
            
            # Predicted return based on ensemble confidence
            if ensemble_pred == 1:
                predicted_return = request.target_growth + (ensemble_prob - 0.5) * 10
            else:
                predicted_return = -(request.target_growth * (1 - ensemble_prob))
            
            target_price = current_price * (1 + predicted_return / 100)
            
            # Step 8: Calculate comprehensive risk score
            risk_score = self._calculate_enhanced_risk_score(
                volatility=volatility,
                technical_indicators=technical_indicators,
                confidence=ensemble_prob,
                rag_insights=rag_insights,
                prediction_breakdown=prediction_breakdown
            )
            
            # Step 9: Generate recommendation
            recommendation = self._generate_enhanced_recommendation(
                prediction=ensemble_pred,
                confidence=ensemble_prob,
                risk_score=risk_score,
                technical_indicators=technical_indicators,
                rag_insights=rag_insights
            )
            
            # Step 10: Calculate stop loss
            stop_loss = self._calculate_stop_loss(
                current_price=current_price,
                atr=technical_indicators.atr,
                volatility=volatility
            )
            
            # Step 11: Generate comprehensive reasons and warnings
            reasons, warnings = self._generate_enhanced_insights(
                prediction=ensemble_pred,
                confidence=ensemble_prob,
                technical_indicators=technical_indicators,
                risk_score=risk_score,
                df=df,
                rag_insights=rag_insights,
                prediction_breakdown=prediction_breakdown,
                trading_strategy=trading_strategy
            )
            
            # Step 12: Calculate sentiment score from knowledge base
            sentiment_score = self._calculate_sentiment_score(rag_insights)
            
            # Create prediction result
            prediction_result = PredictionResult(
                will_grow_10_percent=(ensemble_pred == 1),
                confidence=ensemble_prob,
                predicted_return=predicted_return,
                risk_score=risk_score,
                volatility=volatility,
                recommendation=recommendation
            )
            
            # Create comprehensive response
            response = PredictionResponse(
                symbol=request.symbol,
                exchange=request.exchange,
                prediction=prediction_result,
                technical_indicators=technical_indicators,
                fundamental_score=None,  # Can be enhanced with fundamental data
                sentiment_score=sentiment_score,
                current_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                analysis_timestamp=datetime.now(),
                reasons=reasons,
                warnings=warnings
            )
            
            logger.info(f"✓ Prediction complete: {recommendation} (confidence: {ensemble_prob:.2%})")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    async def _get_historical_data(self, request: PredictionRequest) -> List[CandleData]:
        """Get historical data from request or fetch from API"""
        
        if request.historical_data and len(request.historical_data) > 0:
            logger.info(f"Using provided historical data: {len(request.historical_data)} candles")
            return request.historical_data
        
        logger.info(f"Fetching historical data for {request.symbol}")
        candles = self.data_service.fetch_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            days=365
        )
        
        if not candles or len(candles) == 0:
            raise ValueError(f"Could not fetch historical data for {request.symbol}")
        
        return candles
    
    def _calculate_enhanced_risk_score(
        self,
        volatility: float,
        technical_indicators: TechnicalIndicators,
        confidence: float,
        rag_insights: Dict,
        prediction_breakdown: Dict
    ) -> float:
        """Calculate comprehensive risk score using all available information"""
        
        risk_components = []
        
        # 1. Volatility risk (30% weight)
        volatility_risk = min(1.0, volatility / 0.05)
        risk_components.append(('volatility', volatility_risk, 0.30))
        
        # 2. Technical indicator risk (25% weight)
        rsi = technical_indicators.rsi
        rsi_risk = abs(rsi - 50) / 50 if rsi > 70 or rsi < 30 else 0.3
        risk_components.append(('technical', rsi_risk, 0.25))
        
        # 3. Model confidence risk (25% weight)
        confidence_risk = 1 - confidence
        risk_components.append(('confidence', confidence_risk, 0.25))
        
        # 4. Historical pattern risk (20% weight)
        pattern_risk = 0.5  # Default
        if rag_insights.get('pattern_insights'):
            bearish_count = sum(
                1 for p in rag_insights['pattern_insights']
                if p.get('outcome') == 'bearish'
            )
            total_patterns = len(rag_insights['pattern_insights'])
            pattern_risk = bearish_count / total_patterns if total_patterns > 0 else 0.5
        risk_components.append(('patterns', pattern_risk, 0.20))
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in risk_components)
        risk_score = sum(risk * weight for _, risk, weight in risk_components) / total_weight
        
        return min(1.0, max(0.0, risk_score))
    
    def _generate_enhanced_recommendation(
        self,
        prediction: int,
        confidence: float,
        risk_score: float,
        technical_indicators: TechnicalIndicators,
        rag_insights: Dict
    ) -> str:
        """Generate trading recommendation using ensemble and RAG insights"""
        
        # Strong buy conditions
        if (prediction == 1 and 
            confidence >= 0.75 and 
            risk_score < 0.4 and
            technical_indicators.rsi < 70):
            return "STRONG BUY"
        
        # Buy conditions
        elif (prediction == 1 and 
              confidence >= 0.65 and 
              risk_score < 0.6):
            return "BUY"
        
        # Strong sell conditions
        elif (prediction == 0 and 
              confidence >= 0.75 and
              technical_indicators.rsi > 70):
            return "STRONG SELL"
        
        # Sell conditions
        elif (prediction == 0 and 
              confidence >= 0.65):
            return "SELL"
        
        # Default to hold
        else:
            return "HOLD"
    
    def _calculate_stop_loss(
        self,
        current_price: float,
        atr: float,
        volatility: float
    ) -> float:
        """Calculate intelligent stop loss"""
        
        # Use ATR-based stop loss
        atr_multiplier = 2.0
        atr_stop = current_price - (atr * atr_multiplier)
        
        # Use volatility-based stop loss
        vol_stop = current_price * (1 - 3 * volatility)
        
        # Use the more conservative (higher) stop loss
        stop_loss = max(atr_stop, vol_stop)
        
        # Ensure at least 5% stop loss
        min_stop = current_price * 0.95
        stop_loss = max(stop_loss, min_stop)
        
        return stop_loss
    
    def _generate_enhanced_insights(
        self,
        prediction: int,
        confidence: float,
        technical_indicators: TechnicalIndicators,
        risk_score: float,
        df: pd.DataFrame,
        rag_insights: Dict,
        prediction_breakdown: Dict,
        trading_strategy: Dict
    ) -> tuple:
        """Generate comprehensive reasons and warnings using all available data"""
        
        reasons = []
        warnings = []
        
        # === MODEL INSIGHTS ===
        ensemble_conf = prediction_breakdown['ensemble_probability']
        reasons.append(f"Ensemble model confidence: {ensemble_conf:.1%}")
        
        # Individual model insights
        for model_name, pred_data in prediction_breakdown['base_predictions'].items():
            prob = pred_data.get('probability', 0)
            if prob:
                reasons.append(
                    f"{model_name.upper()} model: {prob:.1%} confidence"
                )
        
        # === TECHNICAL INSIGHTS ===
        rsi = technical_indicators.rsi
        if rsi < 30:
            reasons.append("🔵 RSI indicates oversold condition (strong buying opportunity)")
        elif rsi > 70:
            warnings.append("🔴 RSI indicates overbought condition (potential correction)")
        elif 40 <= rsi <= 60:
            reasons.append("✓ RSI in neutral zone (healthy momentum)")
        
        if technical_indicators.macd > technical_indicators.macd_signal:
            reasons.append("📈 MACD bullish crossover detected")
        elif technical_indicators.macd < technical_indicators.macd_signal:
            warnings.append("📉 MACD bearish crossover detected")
        
        current_price = df['close'].iloc[-1]
        if current_price > technical_indicators.sma_50:
            reasons.append(f"✓ Price above 50-day SMA (₹{technical_indicators.sma_50:.2f}) - uptrend")
        else:
            warnings.append(f"⚠ Price below 50-day SMA (₹{technical_indicators.sma_50:.2f}) - downtrend")
        
        # === KNOWLEDGE BASE INSIGHTS ===
        # Pattern insights
        if rag_insights.get('pattern_insights'):
            bullish_patterns = [p for p in rag_insights['pattern_insights'] if p.get('outcome') == 'bullish']
            bearish_patterns = [p for p in rag_insights['pattern_insights'] if p.get('outcome') == 'bearish']
            
            if len(bullish_patterns) > len(bearish_patterns):
                reasons.append(
                    f"📊 Historical analysis: {len(bullish_patterns)} bullish vs {len(bearish_patterns)} bearish patterns"
                )
            elif len(bearish_patterns) > len(bullish_patterns):
                warnings.append(
                    f"📊 Historical analysis: {len(bearish_patterns)} bearish vs {len(bullish_patterns)} bullish patterns"
                )
        
        # Book insights
        if rag_insights.get('book_insights'):
            strategy_books = [
                b for b in rag_insights['book_insights']
                if b.get('category') in ['trading_strategy', 'technical_analysis']
            ]
            if strategy_books:
                reasons.append(
                    f"📚 {len(strategy_books)} relevant trading strategies from knowledge base"
                )
        
        # Company insights
        if rag_insights.get('company_insights'):
            recent_reports = [
                r for r in rag_insights['company_insights']
                if r.get('year', 0) >= datetime.now().year - 2
            ]
            if recent_reports:
                reasons.append(
                    f"📄 Analysis includes {len(recent_reports)} recent company reports"
                )
        
        # === TRADING STRATEGY INSIGHTS ===
        if trading_strategy.get('entry_criteria'):
            reasons.append(
                f"✓ Found {len(trading_strategy['entry_criteria'])} entry criteria from expert strategies"
            )
        
        if trading_strategy.get('risk_management'):
            reasons.append(
                f"🛡️ {len(trading_strategy['risk_management'])} risk management guidelines available"
            )
        
        # === RISK WARNINGS ===
        if risk_score > 0.7:
            warnings.append(
                f"⚠️ HIGH RISK: Risk score {risk_score:.1%} - significant uncertainty detected"
            )
        
        if confidence < 0.6:
            warnings.append(
                f"⚠️ LOW CONFIDENCE: Model uncertainty is high ({confidence:.1%}) - exercise caution"
            )
        
        # Volume analysis
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].mean()
            if recent_volume > avg_volume * 1.5:
                reasons.append("📊 High trading volume - strong market interest")
            elif recent_volume < avg_volume * 0.5:
                warnings.append("⚠️ Low trading volume - weak market participation")
        
        # === RECOMMENDATIONS ===
        if rag_insights.get('recommendations'):
            for rec in rag_insights['recommendations'][:3]:  # Top 3
                reasons.append(f"💡 {rec}")
        
        return reasons, warnings
    
    def _calculate_sentiment_score(self, rag_insights: Dict) -> float:
        """Calculate sentiment score from RAG insights"""
        
        sentiment_score = 0.5  # Neutral default
        
        # Analyze patterns
        if rag_insights.get('pattern_insights'):
            bullish_count = sum(
                1 for p in rag_insights['pattern_insights']
                if p.get('outcome') == 'bullish'
            )
            total_patterns = len(rag_insights['pattern_insights'])
            
            if total_patterns > 0:
                pattern_sentiment = bullish_count / total_patterns
                sentiment_score = (sentiment_score + pattern_sentiment) / 2
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, sentiment_score))
    
    async def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        
        model_info = {
            'ensemble_trained': self.ensemble_model.is_trained,
            'base_models': {},
            'knowledge_base_stats': {}
        }
        
        # Base model info
        for model_name, model in self.ensemble_model.base_models.items():
            model_info['base_models'][model_name] = {
                'trained': model.is_trained,
                'metrics': model.metrics if model.is_trained else {}
            }
        
        # Knowledge base stats
        try:
            kb_stats = self.rag_service.vector_store.get_collection_stats()
            model_info['knowledge_base_stats'] = kb_stats
        except Exception as e:
            logger.error(f"Error getting KB stats: {str(e)}")
        
        return model_info
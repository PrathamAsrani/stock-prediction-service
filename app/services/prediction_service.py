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
from app.models.ml_model import StockPredictionModel
from app.models.technical_analysis import TechnicalAnalyzer
from app.models.fundamental_analysis import FundamentalAnalyzer
from app.services.data_service import DataService

logger = logging.getLogger(__name__)

class StockPredictionService:
    """Service for stock prediction and analysis"""
    
    def __init__(self):
        self.model = StockPredictionModel(model_type="xgboost")
        self.data_service = DataService()
        self.fundamental_analyzer = FundamentalAnalyzer()
        
        # Try to load pre-trained model
        try:
            self.model.load_model()
            logger.info("Pre-trained model loaded successfully")
        except FileNotFoundError:
            logger.warning("No pre-trained model found. Model needs to be trained first.")
    
    async def predict_stock_growth(self, request: PredictionRequest) -> PredictionResponse:
        """
        Main prediction function
        
        Analyzes stock and predicts if it will grow by target percentage in given days
        """
        logger.info(f"Starting prediction for {request.symbol}")
        
        # Fetch historical data if not provided
        if request.historical_data is None or len(request.historical_data) == 0:
            logger.info(f"Fetching historical data for {request.symbol}")
            candles = self.data_service.fetch_historical_data(
                symbol=request.symbol,
                exchange=request.exchange,
                days=365
            )
            
            if candles is None or len(candles) == 0:
                raise ValueError(f"Could not fetch historical data for {request.symbol}")
        else:
            candles = request.historical_data
        
        # Convert to DataFrame
        df = self.data_service.candles_to_dataframe(candles)
        
        # Calculate technical indicators
        logger.info("Calculating technical indicators")
        technical_indicators = TechnicalAnalyzer.calculate_indicators(candles)
        
        # Add features to dataframe
        df = TechnicalAnalyzer.calculate_features(df)
        
        # Fetch fundamental data if not provided
        fundamental_data = request.fundamental_data
        if fundamental_data is None:
            logger.info("Fetching fundamental data")
            fundamental_data = self.fundamental_analyzer.fetch_fundamentals(
                request.symbol,
                request.exchange
            )
        
        # Calculate fundamental score
        fundamental_score = None
        if fundamental_data is not None:
            fundamental_score = self.fundamental_analyzer.calculate_fundamental_score(fundamental_data)
            logger.info(f"Fundamental score: {fundamental_score:.4f}")
            
            # Add fundamental features to dataframe
            df['pe_ratio'] = fundamental_data.pe_ratio
            df['pb_ratio'] = fundamental_data.pb_ratio
            df['roe'] = fundamental_data.roe
            df['debt_to_equity'] = fundamental_data.debt_to_equity
            df['dividend_yield'] = fundamental_data.dividend_yield
        
        # Make ML prediction
        logger.info("Making ML prediction")
        features = self.model.prepare_features(df)
        prediction, confidence = self.model.predict(features.tail(1))
        
        # Calculate additional metrics
        current_price = df['close'].iloc[-1]
        volatility = df['volatility'].iloc[-1]
        
        # Calculate predicted return and target price
        if prediction == 1:
            predicted_return = request.target_growth + (confidence - 0.5) * 5  # Adjust based on confidence
        else:
            predicted_return = -(request.target_growth * (1 - confidence))
        
        target_price = current_price * (1 + predicted_return / 100)
        
        # Calculate risk score (0-1, higher = riskier)
        risk_score = self._calculate_risk_score(
            volatility=volatility,
            technical_indicators=technical_indicators,
            fundamental_score=fundamental_score,
            confidence=confidence
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            prediction=prediction,
            confidence=confidence,
            risk_score=risk_score,
            technical_indicators=technical_indicators
        )
        
        # Calculate stop loss (5% below current price or based on ATR)
        atr_multiplier = 2.0
        stop_loss = current_price - (technical_indicators.atr * atr_multiplier)
        stop_loss = max(stop_loss, current_price * 0.95)  # At least 5% stop loss
        
        # Generate reasons and warnings
        reasons, warnings = self._generate_insights(
            prediction=prediction,
            confidence=confidence,
            technical_indicators=technical_indicators,
            fundamental_score=fundamental_score,
            risk_score=risk_score,
            df=df
        )
        
        # Create prediction result
        prediction_result = PredictionResult(
            will_grow_10_percent=(prediction == 1),
            confidence=confidence,
            predicted_return=predicted_return,
            risk_score=risk_score,
            volatility=volatility,
            recommendation=recommendation
        )
        
        # Create response
        response = PredictionResponse(
            symbol=request.symbol,
            exchange=request.exchange,
            prediction=prediction_result,
            technical_indicators=technical_indicators,
            fundamental_score=fundamental_score,
            sentiment_score=None,  # Can be added later with news sentiment analysis
            current_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            analysis_timestamp=datetime.now(),
            reasons=reasons,
            warnings=warnings
        )
        
        logger.info(f"Prediction complete for {request.symbol}: {recommendation}")
        
        return response
    
    def _calculate_risk_score(
        self,
        volatility: float,
        technical_indicators: TechnicalIndicators,
        fundamental_score: Optional[float],
        confidence: float
    ) -> float:
        """Calculate overall risk score (0-1)"""
        
        risk_components = []
        
        # Volatility risk (higher volatility = higher risk)
        volatility_risk = min(1.0, volatility / 0.05)  # Normalize around 5% volatility
        risk_components.append(('volatility', volatility_risk, 0.3))
        
        # RSI risk (extreme values = higher risk)
        rsi = technical_indicators.rsi
        if rsi > 70 or rsi < 30:
            rsi_risk = abs(rsi - 50) / 50
        else:
            rsi_risk = 0.3
        risk_components.append(('rsi', rsi_risk, 0.15))
        
        # Confidence risk (lower confidence = higher risk)
        confidence_risk = 1 - confidence
        risk_components.append(('confidence', confidence_risk, 0.25))
        
        # Fundamental risk (if available)
        if fundamental_score is not None:
            fundamental_risk = 1 - fundamental_score
            risk_components.append(('fundamental', fundamental_risk, 0.30))
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in risk_components)
        risk_score = sum(risk * weight for _, risk, weight in risk_components) / total_weight
        
        return min(1.0, max(0.0, risk_score))
    
    def _generate_recommendation(
        self,
        prediction: int,
        confidence: float,
        risk_score: float,
        technical_indicators: TechnicalIndicators
    ) -> str:
        """Generate BUY/HOLD/SELL recommendation"""
        
        if prediction == 1 and confidence >= 0.7 and risk_score < 0.5:
            return "BUY"
        elif prediction == 1 and confidence >= 0.6:
            return "HOLD"
        elif prediction == 0 and confidence >= 0.7:
            return "SELL"
        else:
            return "HOLD"
    
    def _generate_insights(
        self,
        prediction: int,
        confidence: float,
        technical_indicators: TechnicalIndicators,
        fundamental_score: Optional[float],
        risk_score: float,
        df: pd.DataFrame
    ) -> tuple:
        """Generate reasons for prediction and risk warnings"""
        
        reasons = []
        warnings = []
        
        # Technical reasons
        if technical_indicators.rsi < 30:
            reasons.append("RSI indicates oversold condition - potential buying opportunity")
        elif technical_indicators.rsi > 70:
            warnings.append("RSI indicates overbought condition - potential correction ahead")
        
        if technical_indicators.macd > technical_indicators.macd_signal:
            reasons.append("MACD crossed above signal line - bullish momentum")
        elif technical_indicators.macd < technical_indicators.macd_signal:
            warnings.append("MACD crossed below signal line - bearish momentum")
        
        current_price = df['close'].iloc[-1]
        if current_price > technical_indicators.sma_50:
            reasons.append("Price above 50-day SMA - uptrend confirmed")
        else:
            warnings.append("Price below 50-day SMA - downtrend signal")
        
        # Fundamental reasons
        if fundamental_score is not None:
            if fundamental_score > 0.7:
                reasons.append("Strong fundamentals support long-term growth")
            elif fundamental_score < 0.4:
                warnings.append("Weak fundamentals may limit growth potential")
        
        # Risk warnings
        if risk_score > 0.7:
            warnings.append("HIGH RISK: Significant volatility and uncertainty detected")
        
        if confidence < 0.6:
            warnings.append("LOW CONFIDENCE: Model uncertainty is high - proceed with caution")
        
        # Volume analysis
        recent_volume = df['volume'].tail(5).mean()
        avg_volume = df['volume'].mean()
        if recent_volume > avg_volume * 1.5:
            reasons.append("High trading volume indicates strong market interest")
        elif recent_volume < avg_volume * 0.5:
            warnings.append("Low trading volume may indicate weak market interest")
        
        return reasons, warnings
    
    async def train_model(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """Train the ML model with historical data"""
        
        logger.info(f"Training model with {len(symbols)} symbols")
        
        all_data = []
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            
            candles = self.data_service.fetch_historical_data(symbol, days=730)  # 2 years
            
            if candles is None or len(candles) == 0:
                logger.warning(f"Skipping {symbol} - no data available")
                continue
            
            df = self.data_service.candles_to_dataframe(candles)
            df = TechnicalAnalyzer.calculate_features(df)
            
            # Fetch fundamentals
            fundamentals = self.fundamental_analyzer.fetch_fundamentals(symbol)
            if fundamentals:
                df['pe_ratio'] = fundamentals.pe_ratio
                df['pb_ratio'] = fundamentals.pb_ratio
                df['roe'] = fundamentals.roe
                df['debt_to_equity'] = fundamentals.debt_to_equity
                df['dividend_yield'] = fundamentals.dividend_yield
            
            all_data.append(df)
        
        if len(all_data) == 0:
            raise ValueError("No training data available")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total training samples: {len(combined_df)}")
        
        # Train model
        metrics = self.model.train(combined_df)
        
        # Save model
        self.model.save_model()
        
        logger.info("Model training complete")
        
        return {
            "status": "success",
            "metrics": metrics,
            "training_symbols": symbols,
            "total_samples": len(combined_df)
        }
    
    async def get_model_metrics(self) -> Dict:
        """Get current model performance metrics"""
        
        if not self.model.is_trained:
            return {
                "status": "not_trained",
                "message": "Model has not been trained yet"
            }
        
        feature_importance = self.model.get_feature_importance()
        
        return {
            "status": "trained",
            "model_type": self.model.model_type,
            "feature_count": len(self.model.feature_columns),
            "top_features": feature_importance.head(10).to_dict('records') if not feature_importance.empty else []
        }
    
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, date

class CandleData(BaseModel):
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

class FundamentalData(BaseModel):
    """Fundamental analysis data"""
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None

class PredictionRequest(BaseModel):
    """Stock analysis request"""
    symbol: str = Field(..., description="Stock symbol (e.g., ITC, RELIANCE)")
    exchange: str = Field(default="NSE", description="Exchange (NSE/BSE)")
    historical_data: Optional[List[CandleData]] = None
    fundamental_data: Optional[FundamentalData] = None
    days_ahead: int = Field(default=30, description="Days to predict ahead")
    target_growth: float = Field(default=10.0, description="Target growth percentage")

class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    volume_sma: float
    atr: float

class PredictionResult(BaseModel):
    """Prediction results"""
    will_grow_10_percent: bool
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    predicted_return: float = Field(..., description="Predicted return percentage")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score 0-1")
    volatility: float
    recommendation: str = Field(..., description="BUY/HOLD/SELL")

class PredictionResponse(BaseModel):
    """Complete prediction response"""
    symbol: str
    exchange: str
    prediction: PredictionResult
    technical_indicators: TechnicalIndicators
    fundamental_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    current_price: float
    target_price: float
    stop_loss: float
    analysis_timestamp: datetime
    reasons: List[str] = Field(default_factory=list, description="Reasons for prediction")
    warnings: List[str] = Field(default_factory=list, description="Risk warnings")

class TrainingRequest(BaseModel):
    """Model training request"""
    symbols: List[str]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    model_type: str = Field(default="xgboost", description="Model type: xgboost/random_forest/lstm")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    timestamp: datetime
    
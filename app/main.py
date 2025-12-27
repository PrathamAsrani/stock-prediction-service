from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import logging

from app.services.prediction_service import StockPredictionService
from app.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    HealthResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock ML Analysis Service",
    description="ML-powered stock analysis and prediction service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Java service URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = StockPredictionService()

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="Stock ML Analysis Service",
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/api/analyze/predict", response_model=PredictionResponse)
async def analyze_stock(request: PredictionRequest):
    """
    Analyze stock and predict 10% growth in 1 month
    
    Args:
        request: Stock analysis request containing symbol, historical data, etc.
    
    Returns:
        PredictionResponse with analysis results
    """
    try:
        logger.info(f"Analyzing stock: {request.symbol}")
        
        # Perform prediction
        result = await prediction_service.predict_stock_growth(request)
        
        logger.info(f"Analysis complete for {request.symbol}: {result.prediction}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing stock {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze/batch")
async def analyze_multiple_stocks(symbols: List[str]):
    """Analyze multiple stocks in batch"""
    try:
        results = []
        for symbol in symbols:
            request = PredictionRequest(symbol=symbol)
            result = await prediction_service.predict_stock_growth(request)
            results.append(result)
        
        return {
            "total": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/api/model/train")
async def train_model(request: TrainingRequest):
    """Train or retrain the ML model"""
    try:
        logger.info(f"Training model with symbols: {request.symbols}")
        
        result = await prediction_service.train_model(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/model/metrics")
async def get_model_metrics():
    """Get current model performance metrics"""
    try:
        metrics = await prediction_service.get_model_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
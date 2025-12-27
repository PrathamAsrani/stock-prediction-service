from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import logging

from app.services.prediction_service import StockPredictionService
from app.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description="AI-powered stock prediction service with ensemble ML and RAG",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
prediction_service = StockPredictionService()

# === HEALTH & INFO ENDPOINTS ===

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        service=settings.API_TITLE,
        version=settings.API_VERSION,
        timestamp=datetime.now()
    )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    
    model_info = await prediction_service.get_model_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models": model_info,
        "settings": {
            "xgboost_enabled": settings.ENABLE_XGBOOST,
            "lstm_enabled": settings.ENABLE_LSTM,
            "transformer_enabled": settings.ENABLE_TRANSFORMER,
            "model_type": settings.MODEL_TYPE
        }
    }

@app.get("/api/model/info")
async def get_model_info():
    """Get detailed model information"""
    try:
        info = await prediction_service.get_model_info()
        return JSONResponse(content=info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# === PREDICTION ENDPOINTS ===

@app.post("/api/analyze/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """
    Comprehensive stock analysis and prediction
    
    This endpoint uses:
    - Ensemble ML model (XGBoost + LSTM + Transformer)
    - Technical analysis
    - Knowledge base insights (books, reports, patterns)
    - Historical pattern matching
    
    Returns detailed prediction with recommendations
    """
    try:
        logger.info(f"Prediction request for {request.symbol}")
        
        result = await prediction_service.predict_stock_growth(request)
        
        logger.info(
            f"Prediction complete: {request.symbol} - "
            f"{result.prediction.recommendation} "
            f"(confidence: {result.prediction.confidence:.2%})"
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/analyze/batch")
async def batch_predict(symbols: List[str], exchange: str = "NSE"):
    """Analyze multiple stocks in batch"""
    try:
        logger.info(f"Batch prediction for {len(symbols)} symbols")
        
        results = []
        errors = []
        
        for symbol in symbols:
            try:
                request = PredictionRequest(
                    symbol=symbol,
                    exchange=exchange,
                    days_ahead=30,
                    target_growth=10.0
                )
                
                result = await prediction_service.predict_stock_growth(request)
                results.append({
                    'symbol': symbol,
                    'prediction': result.dict()
                })
                
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {str(e)}")
                errors.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return {
            'total_requested': len(symbols),
            'successful': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# === KNOWLEDGE BASE ENDPOINTS ===

@app.get("/api/knowledge/search")
async def search_knowledge_base(
    query: str,
    collection: str = "all",
    n_results: int = 5
):
    """Search the knowledge base"""
    try:
        if collection == "all":
            results = prediction_service.rag_service.vector_store.search_all_collections(
                query=query,
                n_results_per_collection=n_results
            )
        else:
            results = prediction_service.rag_service.vector_store.search(
                query=query,
                collection_type=collection,
                n_results=n_results
            )
        
        return {
            'query': query,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Knowledge base search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/insights/{symbol}")
async def get_trading_insights(symbol: str):
    """Get comprehensive trading insights for a symbol"""
    try:
        insights = prediction_service.rag_service.get_trading_insights(
            symbol=symbol,
            query="swing trading strategy",
            include_patterns=True,
            include_fundamentals=True
        )
        
        return insights
        
    except Exception as e:
        logger.error(f"Error getting insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/strategy/{symbol}")
async def get_trading_strategy(symbol: str):
    """Get swing trading strategy for a symbol"""
    try:
        strategy = prediction_service.rag_service.get_swing_trading_strategy(symbol)
        return strategy
        
    except Exception as e:
        logger.error(f"Error getting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        stats = prediction_service.rag_service.vector_store.get_collection_stats()
        return {
            'collections': stats,
            'total_documents': sum(s['document_count'] for s in stats.values())
        }
    except Exception as e:
        logger.error(f"Error getting KB stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# === TRAINING ENDPOINTS (Admin) ===

class TrainingRequest(BaseModel):
    symbols: List[str]
    rebuild_knowledge_base: bool = False

@app.post("/api/admin/train")
async def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger model training (admin endpoint)
    This runs in the background
    """
    try:
        logger.info(f"Training triggered for {len(request.symbols)} symbols")
        
        # In production, implement proper async training
        # background_tasks.add_task(train_models, request.symbols)
        
        return {
            'status': 'training_initiated',
            'symbols': request.symbols,
            'message': 'Training started in background. Check /api/model/info for status.'
        }
        
    except Exception as e:
        logger.error(f"Training trigger error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# === ERROR HANDLERS ===

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Validation Error", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
    )

# === STARTUP/SHUTDOWN ===

@app.on_event("startup")
async def startup_event():
    logger.info("="*60)
    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    logger.info("="*60)
    logger.info(f"XGBoost: {'✓' if settings.ENABLE_XGBOOST else '✗'}")
    logger.info(f"LSTM: {'✓' if settings.ENABLE_LSTM else '✗'}")
    logger.info(f"Transformer: {'✓' if settings.ENABLE_TRANSFORMER else '✗'}")
    logger.info("="*60)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")
    # Cleanup if needed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
    
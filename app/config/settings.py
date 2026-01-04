from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Stock ML Prediction Service"
    API_VERSION: str = "2.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    
    # Model Settings
    MODEL_TYPE: str = "ensemble"  # xgboost, lstm, transformer, ensemble
    ENABLE_XGBOOST: bool = True
    ENABLE_LSTM: bool = True
    ENABLE_TRANSFORMER: bool = False
    
    # Knowledge Base Settings
    KNOWLEDGE_BASE_PATH: str = "knowledge_base_index"
    BOOKS_PATH: str = "data/books"
    REPORTS_PATH: str = "data/reports"
    HISTORICAL_DATA_PATH: str = "data/historical"
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Training Settings
    TRAIN_TEST_SPLIT: float = 0.2
    VALIDATION_SPLIT: float = 0.1
    RANDOM_SEED: int = 42
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    
    # Prediction Settings
    DEFAULT_DAYS_AHEAD: int = 30
    DEFAULT_TARGET_GROWTH: float = 10.0
    CONFIDENCE_THRESHOLD: float = 0.6
    
    # Data Collection
    YEARS_OF_HISTORY: int = 10
    UPDATE_FREQUENCY_DAYS: int = 1
    
    # External APIs (Optional)
    OPENAI_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_KEY: Optional[str] = None
    
    # Database
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600
    
    # Logging
    LOG_LEVEL: str = "INFO"

    # Zerodha Microservice
    JAVA_SERVICE_URL: str = "http://localhost:3000"
    JAVA_SERVICE_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

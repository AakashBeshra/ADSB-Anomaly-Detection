import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ADS-B Anomaly Detection"
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./adsb.db"
    
    # Redis for caching (optional)
    REDIS_URL: Optional[str] = None
    
    # ML Model Paths
    MODEL_PATH: str = "./models"
    SCALER_PATH: str = "./models/scaler.joblib"
    
    # ADS-B Data Source
    ADSB_API_URL: str = "https://opensky-network.org/api/states/all"
    ADSB_UPDATE_INTERVAL: int = 5  # seconds
    
    # Anomaly Detection Threshold
    ANOMALY_THRESHOLD: float = 0.85
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
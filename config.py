"""
Configuration settings for portfolio analysis API.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """
    
    # API Configuration
    api_title: str = "Portfolio Correlation Analysis API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Analysis Parameters
    max_assets_default: int = 50
    min_assets_required: int = 3
    min_observations_multiplier: int = 3  # min_obs = num_assets * this
    default_period: str = "1y"
    supported_periods: List[str] = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    
    # Random Matrix Theory Settings
    mp_q_ratio_threshold: float = 0.5  # Above this, MP analysis less reliable
    mp_confidence_threshold: float = 0.6  # Minimum confidence for MP results
    
    # Data Quality Thresholds
    max_missing_data_pct: float = 0.1  # 10% max missing data
    min_correlation_threshold: float = -0.98  # Avoid perfect negative correlations
    max_correlation_threshold: float = 0.98   # Avoid perfect positive correlations
    
    # Performance Settings
    cache_timeout_seconds: int = 3600  # 1 hour cache for market data
    request_timeout_seconds: int = 30
    max_concurrent_requests: int = 10
    
    # File Upload Settings
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = [".csv", ".xlsx"]
    
    # Database Settings (for future use)
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # External API Settings
    yfinance_timeout: int = 10
    alpha_vantage_api_key: Optional[str] = None  # For alternative data source
    
    # Security Settings
    cors_origins: List[str] = ["*"]  # Configure for production
    api_key_required: bool = False
    rate_limit_per_minute: int = 60
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Report Generation
    max_report_length: int = 10000  # Character limit for reports
    include_technical_details: bool = True
    
    class Config:
        env_file = ".env"
        env_prefix = "PORTFOLIO_"


class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]


class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "WARNING"
    api_key_required: bool = True
    cors_origins: List[str] = []  # Configure with actual frontend domains
    cache_timeout_seconds: int = 7200  # 2 hours in production


class TestingSettings(Settings):
    """Testing environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    max_assets_default: int = 10  # Smaller for faster tests
    cache_timeout_seconds: int = 60  # Short cache for tests
    yfinance_timeout: int = 5  # Faster timeouts for tests


def get_settings() -> Settings:
    """
    Get settings based on environment.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


def get_analysis_config() -> dict:
    """
    Get analysis-specific configuration for the analyzer.
    """
    settings = get_settings()
    
    return {
        "max_assets": settings.max_assets_default,
        "min_observations_multiplier": settings.min_observations_multiplier,
        "mp_q_threshold": settings.mp_q_ratio_threshold,
        "mp_confidence_threshold": settings.mp_confidence_threshold,
        "data_quality": {
            "max_missing_pct": settings.max_missing_data_pct,
            "correlation_bounds": (settings.min_correlation_threshold, settings.max_correlation_threshold)
        }
    }


def get_data_service_config() -> dict:
    """
    Get data service configuration.
    """
    settings = get_settings()
    
    return {
        "timeout": settings.yfinance_timeout,
        "cache_timeout": settings.cache_timeout_seconds,
        "supported_periods": settings.supported_periods,
        "alpha_vantage_key": settings.alpha_vantage_api_key
    }


# Example environment variables for .env file:
ENV_EXAMPLE = """
# Portfolio Analysis API Environment Variables
ENVIRONMENT=development
PORTFOLIO_DEBUG=true
PORTFOLIO_MAX_ASSETS_DEFAULT=50
PORTFOLIO_API_KEY_REQUIRED=false
PORTFOLIO_LOG_LEVEL=INFO
PORTFOLIO_YFINANCE_TIMEOUT=10
PORTFOLIO_CORS_ORIGINS=["http://localhost:3000"]

# For production:
# ENVIRONMENT=production
# PORTFOLIO_DEBUG=false
# PORTFOLIO_API_KEY_REQUIRED=true
# PORTFOLIO_CORS_ORIGINS=["https://yourdomain.com"]
"""

if __name__ == "__main__":
    # Print current configuration
    settings = get_settings()
    print("Current Configuration:")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Debug: {settings.debug}")
    print(f"Max Assets: {settings.max_assets_default}")
    print(f"Default Period: {settings.default_period}")
    print(f"Log Level: {settings.log_level}")
    
    print("\nExample .env file content:")
    print(ENV_EXAMPLE)
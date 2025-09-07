from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import engine
from app.models.portfolio import Base

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Portfolio Analyzer API")
    
    # Create database tables (commented out for now - we'll add DB later)
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Portfolio Analyzer API")

app = FastAPI(
    title=settings.APP_NAME,
    description="Advanced portfolio analysis with institutional-quality risk metrics",
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION
    }

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Portfolio Analyzer API",
        "version": settings.VERSION,
        "docs": "/docs"
    }

# Test endpoint for your analysis
@app.post("/api/v1/analyze")
async def quick_analyze(tickers: List[str]):
    """Quick analysis endpoint for testing"""
    from app.services.data_service import DataService
    from app.services.analysis_service import AnalysisService
    
    data_service = DataService()
    analysis_service = AnalysisService(data_service)
    
    try:
        result = await analysis_service.analyze_portfolio(
            tickers=tickers,
            analysis_types=['basic', 'correlation']
        )
        return result
    except Exception as e:
        return {"error": str(e)}
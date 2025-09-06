# Portfolio Analyzer Backend - Clean Implementation
# Building from scratch with proper structure

# ==============================================================================
# CREATE PROJECT STRUCTURE
# ==============================================================================

echo "Creating backend project structure..."

# Create directory structure
mkdir -p backend/{app/{core,models,schemas,services,routers,utils},tests,alembic}

# Create __init__.py files
touch backend/app/__init__.py
touch backend/app/core/__init__.py
touch backend/app/models/__init__.py
touch backend/app/schemas/__init__.py
touch backend/app/services/__init__.py
touch backend/app/routers/__init__.py
touch backend/app/utils/__init__.py
touch backend/tests/__init__.py

echo "Project structure created!"

# ==============================================================================
# CORE CONFIGURATION
# ==============================================================================

# backend/app/core/config.py
cat > backend/app/core/config.py << 'EOF'
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Portfolio Analyzer"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/portfolio_analyzer"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Analysis
    MAX_PORTFOLIO_SIZE: int = 100
    DEFAULT_START_DATE: str = "2020-01-01"
    CACHE_TTL_SECONDS: int = 3600
    
    class Config:
        env_file = ".env"

settings = Settings()
EOF

# backend/app/core/database.py
cat > backend/app/core/database.py << 'EOF'
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=20,
    max_overflow=0
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
EOF

# ==============================================================================
# DATA MODELS
# ==============================================================================

# backend/app/models/portfolio.py
cat > backend/app/models/portfolio.py << 'EOF'
from sqlalchemy import Column, String, Text, DateTime, JSON, Float, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    portfolios = relationship("Portfolio", back_populates="owner")

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    tickers = Column(JSON, nullable=False)  # ["AAPL", "MSFT", ...]
    weights = Column(JSON)  # [0.5, 0.3, 0.2] or null for equal weight
    benchmark_ticker = Column(String(10), default="SPY")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="portfolios")
    analyses = relationship("Analysis", back_populates="portfolio", cascade="all, delete-orphan")

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False, index=True)
    parameters = Column(JSON)  # Analysis parameters
    results = Column(JSON, nullable=False)  # Analysis results
    computation_time = Column(Float)  # Seconds
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    portfolio = relationship("Portfolio", back_populates="analyses")
EOF

# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

# backend/app/schemas/portfolio.py
cat > backend/app/schemas/portfolio.py << 'EOF'
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class PortfolioBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    tickers: List[str] = Field(..., min_items=1, max_items=100)
    weights: Optional[List[float]] = None
    benchmark_ticker: str = Field(default="SPY", max_length=10)

    @validator('tickers')
    def validate_tickers(cls, v):
        if not v:
            raise ValueError("At least one ticker is required")
        
        cleaned_tickers = []
        for ticker in v:
            clean = ticker.strip().upper()
            if len(clean) > 10 or len(clean) < 1:
                raise ValueError(f"Invalid ticker: {clean}")
            if not clean.replace('.', '').replace('-', '').isalnum():
                raise ValueError(f"Invalid ticker format: {clean}")
            cleaned_tickers.append(clean)
        
        return cleaned_tickers

    @validator('weights')
    def validate_weights(cls, v, values):
        if v is not None:
            tickers = values.get('tickers', [])
            if len(v) != len(tickers):
                raise ValueError("Weights must match number of tickers")
            if any(w < 0 for w in v):
                raise ValueError("Weights cannot be negative")
            if abs(sum(v) - 1.0) > 0.01:
                raise ValueError("Weights must sum to 1.0")
        return v

class PortfolioCreate(PortfolioBase):
    pass

class PortfolioResponse(PortfolioBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class AnalysisRequest(BaseModel):
    portfolio_id: uuid.UUID
    analysis_types: List[str] = Field(default=["basic", "correlation"])
    start_date: Optional[str] = None  # YYYY-MM-DD format
    parameters: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    portfolio_id: uuid.UUID
    analysis_types: List[str]
    computation_time: float
    results: Dict[str, Any]
EOF

# ==============================================================================
# DATA SERVICE (IMPROVED FROM YOUR ARCHIVE)
# ==============================================================================

# backend/app/services/data_service.py
cat > backend/app/services/data_service.py << 'EOF'
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import yfinance as yf
from app.core.config import settings

logger = logging.getLogger(__name__)

class DataService:
    """
    Async data service for fetching financial data
    Start with yfinance, can add other providers later
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.cache_ttl = settings.CACHE_TTL_SECONDS
        
    async def fetch_portfolio_data(
        self,
        tickers: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Fetch price data for portfolio tickers
        Returns: (price_data, failed_tickers, metadata)
        """
        if start_date is None:
            start_date = settings.DEFAULT_START_DATE
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Use thread pool for yfinance (blocking operation)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self._fetch_data_sync, 
            tickers, 
            start_date, 
            end_date
        )
        
        return result
    
    def _fetch_data_sync(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """Synchronous data fetch using yfinance"""
        
        successful_data = {}
        failed_tickers = []
        
        # Process in batches
        batch_size = 20
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            
            try:
                if len(batch) == 1:
                    # Single ticker
                    ticker = batch[0]
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not data.empty and 'Close' in data.columns:
                        successful_data[ticker] = data['Close']
                    else:
                        failed_tickers.append(ticker)
                else:
                    # Multiple tickers
                    data = yf.download(batch, start=start_date, end=end_date, progress=False)
                    if not data.empty and 'Close' in data.columns.get_level_values(0):
                        for ticker in batch:
                            if ticker in data['Close'].columns:
                                series = data['Close'][ticker]
                                if not series.empty and not series.isna().all():
                                    successful_data[ticker] = series
                                else:
                                    failed_tickers.append(ticker)
                            else:
                                failed_tickers.append(ticker)
                    else:
                        failed_tickers.extend(batch)
                        
            except Exception as e:
                logger.warning(f"Batch fetch failed for {batch}: {e}")
                failed_tickers.extend(batch)
        
        if not successful_data:
            raise ValueError("No data fetched for any ticker")
        
        # Create DataFrame
        price_df = pd.DataFrame(successful_data)
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.sort_index()
        
        # Remove rows with too many NaN values
        price_df = price_df.dropna(thresh=len(price_df.columns) * 0.8)
        
        metadata = {
            'start_date': start_date,
            'end_date': end_date,
            'successful_tickers': list(successful_data.keys()),
            'failed_tickers': failed_tickers,
            'total_observations': len(price_df),
            'date_range': {
                'start': price_df.index[0].strftime('%Y-%m-%d') if len(price_df) > 0 else None,
                'end': price_df.index[-1].strftime('%Y-%m-%d') if len(price_df) > 0 else None
            }
        }
        
        return price_df, failed_tickers, metadata
EOF

# ==============================================================================
# ANALYSIS SERVICE (YOUR CORE INNOVATION)
# ==============================================================================

# backend/app/services/analysis_service.py
cat > backend/app/services/analysis_service.py << 'EOF'
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from scipy import stats
from app.services.data_service import DataService

logger = logging.getLogger(__name__)

class AnalysisService:
    """
    Portfolio analysis service with your correlation expertise
    """
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
    
    async def analyze_portfolio(
        self,
        tickers: List[str],
        weights: Optional[List[float]] = None,
        benchmark_ticker: str = "SPY",
        start_date: str = None,
        analysis_types: List[str] = None
    ) -> Dict[str, Any]:
        """Main portfolio analysis pipeline"""
        
        if analysis_types is None:
            analysis_types = ['basic', 'correlation']
        
        # Fetch price data
        price_data, failed_tickers, metadata = await self.data_service.fetch_portfolio_data(
            tickers, start_date
        )
        
        # Update tickers and weights for successful data
        valid_tickers = price_data.columns.tolist()
        if weights:
            # Adjust weights for failed tickers
            original_tickers = [t for t in tickers if t not in failed_tickers]
            weight_mapping = dict(zip(original_tickers, weights))
            valid_weights = [weight_mapping.get(t, 0) for t in valid_tickers]
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()  # Renormalize
        else:
            valid_weights = np.array([1/len(valid_tickers)] * len(valid_tickers))
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        portfolio_returns = (returns * valid_weights).sum(axis=1)
        
        # Build results
        results = {
            'metadata': {
                'portfolio_tickers': valid_tickers,
                'weights': valid_weights.tolist(),
                'failed_tickers': failed_tickers,
                'benchmark_ticker': benchmark_ticker,
                'analysis_period': metadata['date_range'],
                'total_observations': len(returns)
            }
        }
        
        if 'basic' in analysis_types:
            results['basic'] = self._basic_analysis(portfolio_returns)
        
        if 'correlation' in analysis_types:
            results['correlation'] = self._correlation_analysis(returns, valid_weights)
        
        return results
    
    def _basic_analysis(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """Standard portfolio metrics"""
        
        # Annualized metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Risk metrics
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Drawdown analysis
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'total_return': float((1 + portfolio_returns).prod() - 1),
            'win_rate': float((portfolio_returns > 0).mean()),
        }
    
    def _correlation_analysis(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """
        YOUR KEY INNOVATION: Correlation structure analysis
        This is what makes your tool valuable
        """
        
        correlation_matrix = returns.corr().fillna(0)
        np.fill_diagonal(correlation_matrix.values, 1.0)
        
        # Eigenvalue analysis (your expertise from Capital Group)
        try:
            eigenvalues = np.linalg.eigvals(correlation_matrix.values)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        except np.linalg.LinAlgError:
            logger.error("Eigenvalue decomposition failed")
            eigenvalues = np.array([1.0])
        
        # Effective rank calculation (participation ratio)
        eigenvalue_weights = eigenvalues / eigenvalues.sum()
        effective_rank = 1 / np.sum(eigenvalue_weights**2)
        concentration_ratio = effective_rank / len(returns.columns)
        
        # Condition number
        condition_number = np.max(eigenvalues) / np.max([np.min(eigenvalues), 1e-15])
        
        # Portfolio concentration metrics
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix.values, weights))
        average_correlation = (portfolio_variance - 1) / (len(weights) - 1) if len(weights) > 1 else 0
        
        return {
            'effective_rank': float(effective_rank),
            'concentration_ratio': float(concentration_ratio),
            'condition_number': float(condition_number),
            'eigenvalues': eigenvalues.tolist()[:10],  # Top 10 eigenvalues
            'explained_variance_ratios': (eigenvalues / eigenvalues.sum()).tolist()[:10],
            'portfolio_concentration': {
                'average_correlation': float(average_correlation),
                'portfolio_variance': float(portfolio_variance)
            },
            'correlation_statistics': {
                'mean_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()),
                'max_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()),
                'min_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()),
            }
        }
EOF

# ==============================================================================
# FASTAPI MAIN APPLICATION
# ==============================================================================

# backend/app/main.py
cat > backend/app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
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
EOF

# ==============================================================================
# QUICK TEST SCRIPT
# ==============================================================================

# test_backend.py
cat > test_backend.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test of the backend implementation
"""
import asyncio
import sys
sys.path.append('backend')

from app.services.data_service import DataService
from app.services.analysis_service import AnalysisService

async def test_analysis():
    """Test the analysis service"""
    
    # Test portfolio (your retirement-focused stocks)
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'VTI', 'BND']
    print(f"Testing analysis with: {test_tickers}")
    
    # Initialize services
    data_service = DataService()
    analysis_service = AnalysisService(data_service)
    
    try:
        # Run analysis
        result = await analysis_service.analyze_portfolio(
            tickers=test_tickers,
            analysis_types=['basic', 'correlation']
        )
        
        # Print results
        print("\n" + "="*50)
        print("PORTFOLIO ANALYSIS RESULTS")
        print("="*50)
        
        # Metadata
        metadata = result['metadata']
        print(f"\nPortfolio: {', '.join(metadata['portfolio_tickers'])}")
        print(f"Analysis Period: {metadata['analysis_period']['start']} to {metadata['analysis_period']['end']}")
        print(f"Observations: {metadata['total_observations']}")
        
        # Basic metrics
        if 'basic' in result:
            basic = result['basic']
            print(f"\nBASIC METRICS:")
            print(f"  Annual Return: {basic['annual_return']:.1%}")
            print(f"  Annual Volatility: {basic['annual_volatility']:.1%}")
            print(f"  Sharpe Ratio: {basic['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {basic['max_drawdown']:.1%}")
        
        # Correlation analysis (YOUR KEY INSIGHT)
        if 'correlation' in result:
            corr = result['correlation']
            print(f"\nCORRELATION ANALYSIS (Your Innovation):")
            print(f"  Portfolio Assets: {len(metadata['portfolio_tickers'])}")
            print(f"  Effective Risk Factors: {corr['effective_rank']:.1f}")
            print(f"  Concentration Ratio: {corr['concentration_ratio']:.3f}")
            print(f"  Diversification Score: {1 - corr['concentration_ratio']:.3f}")
            
            if corr['concentration_ratio'] < 0.5:
                print(f"  ⚠️  Portfolio is more concentrated than it appears!")
        
        print("\n✅ Analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_analysis())
    sys.exit(0 if success else 1)
EOF

chmod +x test_backend.py

# ==============================================================================
# SETUP INSTRUCTIONS
# ==============================================================================

echo ""
echo "==================================================================="
echo "BACKEND CREATED! Next steps:"
echo "==================================================================="
echo ""
echo "1. Test the backend:"
echo "   python test_backend.py"
echo ""
echo "2. Start the FastAPI server:"
echo "   cd backend"
echo "   uv run uvicorn app.main:app --reload"
echo ""
echo "3. Check the API docs:"
echo "   http://localhost:8000/docs"
echo ""
echo "4. Test quick analysis endpoint:"
echo "   curl -X POST \"http://localhost:8000/api/v1/analyze\" \\"
echo "        -H \"Content-Type: application/json\" \\"
echo "        -d '[\"AAPL\", \"MSFT\", \"GOOGL\"]'"
echo ""
echo "==================================================================="
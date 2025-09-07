"""
FastAPI endpoints for portfolio correlation analysis.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import io
import sys
import os

# Add both src and project root to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from analysis.portfolio_analyzer import PortfolioCorrelationAnalyzer, analyze_returns_data
from data.data_service import DataService, load_sample_portfolio_data
from config import get_settings

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio Correlation Analysis API",
    description="Advanced portfolio diversification analysis using eigenvalue decomposition and Random Matrix Theory",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = get_settings()
analyzer = PortfolioCorrelationAnalyzer()
data_service = DataService()

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    tickers: List[str]
    weights: Optional[List[float]] = None
    period: str = "1y"
    max_assets: Optional[int] = None

class AnalysisResponse(BaseModel):
    analysis_successful: bool
    error: Optional[str] = None
    effective_rank: Optional[float] = None
    diversification_loss: Optional[float] = None
    num_assets: Optional[int] = None
    mp_fitting_successful: Optional[bool] = None
    noise_fraction: Optional[float] = None
    num_signal_factors: Optional[int] = None
    interpretation: Optional[List[str]] = None
    advisor_talking_points: Optional[List[str]] = None
    quality_metrics: Optional[Dict[str, Any]] = None

class ReportRequest(BaseModel):
    report_type: str = "comprehensive"  # "summary", "advisor", "comprehensive"

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "portfolio-analysis"}

@app.get("/test")
async def simple_test():
    """Simple test endpoint"""
    return {"status": "working", "message": "API is responding"}

@app.get("/test/data")
async def test_data_loading():
    """Test data loading without analysis"""
    try:
        from data.data_service import load_sample_portfolio_data
        returns_df, weights = load_sample_portfolio_data("sample")
        
        return {
            "status": "success",
            "num_assets": len(returns_df.columns),
            "num_observations": len(returns_df),
            "assets": returns_df.columns.tolist()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
@app.get("/test/analysis")
async def test_analysis():
    """Test analysis with synthetic data"""
    try:
        import numpy as np
        import pandas as pd
        
        # Create simple synthetic data
        np.random.seed(42)
        data = np.random.normal(0, 0.01, (100, 5))
        returns_df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
        
        from analysis.portfolio_analyzer import PortfolioCorrelationAnalyzer
        analyzer = PortfolioCorrelationAnalyzer()
        results = analyzer.analyze_portfolio_returns(returns_df)
        
        return {
            "status": "success",
            "effective_rank": float(results.get('effective_rank', 0)),
            "analysis_successful": results.get('analysis_successful', False)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
@app.get("/test/analysis-debug")
async def test_analysis_debug():
    """Test analysis with detailed error reporting"""
    try:
        # Use the same sample data that worked
        from data.data_service import load_sample_portfolio_data
        returns_df, weights = load_sample_portfolio_data("sample")
        
        from analysis.portfolio_analyzer import PortfolioCorrelationAnalyzer
        analyzer = PortfolioCorrelationAnalyzer()
        results = analyzer.analyze_portfolio_returns(returns_df, weights)
        
        # Return detailed results for debugging
        return {
            "status": "success",
            "analysis_successful": results.get('analysis_successful', False),
            "error": results.get('error', None),
            "effective_rank": results.get('effective_rank', None),
            "num_assets": results.get('num_assets', None),
            "mp_fitting_successful": results.get('mp_fitting_successful', None),
            "all_keys": list(results.keys())
        }
    except Exception as e:
        import traceback
        return {
            "status": "error", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Main analysis endpoints
@app.post("/analyze/tickers", response_model=AnalysisResponse)
async def analyze_ticker_portfolio(request: AnalysisRequest):
    """
    Analyze portfolio correlation structure from ticker list.
    """
    try:
        # Validate input
        if not request.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        
        if len(request.tickers) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 tickers for meaningful analysis")
        
        # Fetch returns data
        returns_df = data_service.fetch_returns_data(
            request.tickers, 
            period=request.period
        )
        
        # Create weights if provided
        weights_series = None
        if request.weights:
            if len(request.weights) != len(returns_df.columns):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Weights length ({len(request.weights)}) doesn't match available tickers ({len(returns_df.columns)})"
                )
            weights_series = pd.Series(request.weights, index=returns_df.columns)
            weights_series = weights_series / weights_series.sum()  # Normalize
        
        # Limit assets if requested
        if request.max_assets and len(returns_df.columns) > request.max_assets:
            if weights_series is not None:
                # Keep top weighted assets
                top_assets = weights_series.nlargest(request.max_assets).index
                returns_df = returns_df[top_assets]
                weights_series = weights_series[top_assets]
            else:
                # Keep first N assets
                returns_df = returns_df.iloc[:, :request.max_assets]
        
        # Run analysis
        results = analyzer.analyze_portfolio_returns(returns_df, weights_series)
        
        # Convert to response model
        response = AnalysisResponse(**results)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/csv")
async def analyze_csv_portfolio(
    file: UploadFile = File(...),
    ticker_column: str = Query("Ticker", description="Name of ticker column"),
    weight_column: str = Query("Portfolio Weight", description="Name of weight column"),
    period: str = Query("1y", description="Time period for returns"),
    max_assets: Optional[int] = Query(None, description="Maximum number of assets to analyze")
):
    """
    Analyze portfolio correlation structure from uploaded CSV file.
    """
    try:
        # Read CSV file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate columns
        if ticker_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{ticker_column}' not found in CSV")
        
        # Clean and extract data
        tickers = df[ticker_column].dropna().astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t and t != 'nan']
        
        if len(tickers) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 valid tickers")
        
        weights = None
        if weight_column in df.columns:
            weights_data = pd.to_numeric(df[weight_column], errors='coerce').dropna()
            if len(weights_data) > 0:
                # Align weights with tickers
                ticker_weight_df = df[[ticker_column, weight_column]].dropna()
                weights_dict = dict(zip(ticker_weight_df[ticker_column], ticker_weight_df[weight_column]))
                weights = [weights_dict.get(ticker, 0) for ticker in tickers]
        
        # Create analysis request
        request = AnalysisRequest(
            tickers=tickers,
            weights=weights,
            period=period,
            max_assets=max_assets
        )
        
        # Use existing ticker analysis endpoint
        return await analyze_ticker_portfolio(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV analysis failed: {str(e)}")

@app.get("/analyze/sample")
async def analyze_sample_portfolio():
    """
    Analyze sample portfolio for testing/demonstration.
    """
    try:
        # Load sample data
        returns_df, weights = load_sample_portfolio_data("sample")
        
        # Run analysis
        results = analyzer.analyze_portfolio_returns(returns_df, weights)
        
        if not results.get('analysis_successful', False):
            raise HTTPException(status_code=500, detail=f"Analysis failed: {results.get('error', 'Unknown error')}")
        
        # Convert numpy types and complex data to JSON-serializable format
        def convert_for_json(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            else:
                return obj
        
        # Convert all values
        json_results = convert_for_json(results)
        
        # Create response with only the fields that the Pydantic model expects
        response_data = {
            'analysis_successful': json_results.get('analysis_successful', False),
            'effective_rank': json_results.get('effective_rank'),
            'diversification_loss': json_results.get('diversification_loss'),
            'num_assets': json_results.get('num_assets'),
            'mp_fitting_successful': json_results.get('mp_fitting_successful'),
            'noise_fraction': json_results.get('noise_fraction'),
            'num_signal_factors': json_results.get('num_signal_factors'),
            'interpretation': json_results.get('interpretation'),
            'advisor_talking_points': json_results.get('advisor_talking_points'),
            'quality_metrics': json_results.get('quality_metrics')
        }
        
        return response_data  # Return dict directly instead of Pydantic model
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample analysis failed: {str(e)}")

# Report generation endpoints
@app.post("/report/{analysis_id}")
async def generate_report(analysis_id: str, request: ReportRequest):
    """
    Generate formatted report from previous analysis.
    Note: In production, you'd store analysis results and retrieve by ID.
    For now, this is a placeholder for the report generation functionality.
    """
    # This would typically retrieve stored analysis results by ID
    # For demo purposes, return sample report structure
    return {
        "analysis_id": analysis_id,
        "report_type": request.report_type,
        "message": "Report generation endpoint - implement with persistent storage"
    }

# Utility endpoints
@app.get("/validate/tickers")
async def validate_tickers(tickers: str = Query(..., description="Comma-separated list of tickers")):
    """
    Validate that tickers can be retrieved from data source.
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        validation_results = data_service.validate_ticker_data(ticker_list)
        
        valid_count = sum(validation_results.values())
        
        return {
            "total_tickers": len(ticker_list),
            "valid_tickers": valid_count,
            "invalid_tickers": len(ticker_list) - valid_count,
            "validation_details": validation_results,
            "success_rate": valid_count / len(ticker_list) if ticker_list else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/data/quality")
async def get_data_quality_info(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
    period: str = Query("1y", description="Time period for data quality check")
):
    """
    Get data quality information for given tickers.
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        
        # Fetch data
        returns_df = data_service.fetch_returns_data(ticker_list, period=period)
        
        # Generate quality report
        quality_report = data_service.get_data_quality_report(returns_df)
        
        return quality_report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data quality check failed: {str(e)}")

# Configuration endpoints
@app.get("/config")
async def get_configuration():
    """
    Get current API configuration.
    """
    return {
        "api_version": "1.0.0",
        "max_assets_default": settings.max_assets_default,
        "default_period": settings.default_period,
        "supported_periods": settings.supported_periods,
        "analysis_features": {
            "effective_rank": True,
            "marchenko_pastur": True,
            "risk_concentration": True,
            "advisor_reports": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
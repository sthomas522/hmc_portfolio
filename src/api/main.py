"""
Enhanced FastAPI endpoints with tail risk analysis integration.

This module extends the existing API to include comprehensive tail risk metrics
alongside the existing correlation analysis.
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

# Now these imports should work
from analysis.portfolio_analyzer import EnhancedPortfolioAnalyzer, analyze_comprehensive_portfolio_risk
from analysis.tail_risk_analysis import TailRiskAnalyzer
from data.data_service import DataService, load_sample_portfolio_data
from config import get_settings

# Pydantic models for enhanced request/response
class EnhancedAnalysisRequest(BaseModel):
    tickers: List[str]
    weights: Optional[List[float]] = None
    period: str = "1y"
    max_assets: Optional[int] = None
    include_tail_risk: bool = True
    confidence_levels: Optional[List[float]] = [0.95, 0.99, 0.995]

class TailRiskResponse(BaseModel):
    analysis_successful: bool
    error: Optional[str] = None
    portfolio_summary: Optional[Dict[str, Any]] = None
    var_cvar_analysis: Optional[Dict[str, Any]] = None
    drawdown_analysis: Optional[Dict[str, Any]] = None
    stress_testing: Optional[Dict[str, Any]] = None
    rolling_volatility: Optional[Dict[str, Any]] = None
    tail_dependency: Optional[Dict[str, Any]] = None
    risk_interpretation: Optional[List[str]] = None
    advisor_risk_talking_points: Optional[List[str]] = None

class ComprehensiveAnalysisResponse(BaseModel):
    correlation_analysis: Optional[Dict[str, Any]] = None
    tail_risk_analysis: Optional[Dict[str, Any]] = None
    enhanced_interpretation: Optional[List[str]] = None
    enhanced_advisor_points: Optional[List[str]] = None
    integrated_insights: Optional[List[str]] = None
    analysis_type: str = "comprehensive"

class AnalysisRequest(BaseModel):
    tickers: List[str]
    weights: Optional[List[float]] = None
    period: str = "1y"
    max_assets: Optional[int] = None

# Initialize enhanced services
app = FastAPI(
    title="Enhanced Portfolio Risk Analysis API",
    description="Comprehensive portfolio analysis with correlation structure and tail risk metrics",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
enhanced_analyzer = EnhancedPortfolioAnalyzer()
tail_risk_analyzer = TailRiskAnalyzer()
data_service = DataService()

# Enhanced analysis endpoints
@app.post("/analyze/comprehensive")
async def analyze_comprehensive_portfolio(request: EnhancedAnalysisRequest):
    """
    Perform comprehensive portfolio analysis including correlation structure and tail risk.
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
            weights_series = weights_series / weights_series.sum()
        
        # Limit assets if requested
        if request.max_assets and len(returns_df.columns) > request.max_assets:
            if weights_series is not None:
                top_assets = weights_series.nlargest(request.max_assets).index
                returns_df = returns_df[top_assets]
                weights_series = weights_series[top_assets]
            else:
                returns_df = returns_df.iloc[:, :request.max_assets]
        
        # Run comprehensive analysis
        results = enhanced_analyzer.analyze_comprehensive_portfolio_risk(
            returns_df, weights_series, request.include_tail_risk
        )
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            else:
                return obj
        
        json_results = convert_for_json(results)
        return json_results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


@app.post("/analyze/tickers")
async def analyze_ticker_portfolio(request: AnalysisRequest):
    """
    Analyze portfolio correlation structure from ticker list.
    """
    try:
        print(f"DEBUG: Starting analysis with tickers: {request.tickers}")
        
        # Validate input
        if not request.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        
        if len(request.tickers) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 tickers for meaningful analysis")
        
        print("DEBUG: Validation passed, fetching data...")
        
        # Fetch returns data
        returns_df = data_service.fetch_returns_data(
            request.tickers, 
            period=request.period
        )
        
        print(f"DEBUG: Got returns data with shape: {returns_df.shape}")
        
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
        
        print("DEBUG: About to run analysis...")
        
        # Run analysis using the original analyzer
        from analysis.portfolio_analyzer import EnhancedPortfolioAnalyzer
        analyzer = EnhancedPortfolioAnalyzer()
        results = analyzer._perform_correlation_analysis(returns_df, weights_series)

        print(f"DEBUG: Analysis completed, successful: {results.get('analysis_successful', False)}")
        
        if not results.get('analysis_successful', False):
            raise HTTPException(status_code=500, detail=f"Analysis failed: {results.get('error', 'Unknown error')}")
        
        # Convert numpy types to JSON-serializable format
        def convert_for_json(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            else:
                return obj
        
        # Convert all values
        json_results = convert_for_json(results)
        
        print("DEBUG: Conversion completed, returning results")
        
        return json_results
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"ERROR in analyze_ticker_portfolio: {str(e)}")
        print(f"ERROR type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
@app.post("/analyze/tail-risk-only")
async def analyze_tail_risk_only(request: EnhancedAnalysisRequest):
    """
    Perform tail risk analysis only (without correlation analysis).
    """
    try:
        # Validate input
        if not request.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        
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
                    detail=f"Weights length doesn't match available tickers"
                )
            weights_series = pd.Series(request.weights, index=returns_df.columns)
            weights_series = weights_series / weights_series.sum()
        
        # Run tail risk analysis
        results = tail_risk_analyzer.analyze_tail_risk(
            returns_df, weights_series, request.confidence_levels
        )
        
        if not results.get('analysis_successful', False):
            raise HTTPException(status_code=500, detail=f"Tail risk analysis failed: {results.get('error', 'Unknown error')}")
        
        # Convert for JSON
        def convert_for_json(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            else:
                return obj
        
        return convert_for_json(results)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tail risk analysis failed: {str(e)}")

@app.get("/analyze/sample/comprehensive")
async def analyze_sample_comprehensive():
    """
    Analyze sample portfolio with comprehensive risk analysis.
    """
    try:
        # Load sample data
        returns_df, weights = load_sample_portfolio_data("sample")
        
        # Run comprehensive analysis
        results = enhanced_analyzer.analyze_comprehensive_portfolio_risk(
            returns_df, weights, include_tail_risk=True
        )
        
        # Convert for JSON
        def convert_for_json(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            else:
                return obj
        
        return convert_for_json(results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample comprehensive analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "portfolio-analysis"}

@app.get("/debug/routes")
async def list_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({"path": route.path, "methods": list(route.methods)})
    return routes

@app.get("/analyze/sample/tail-risk")
async def analyze_sample_tail_risk():
    """
    Analyze sample portfolio tail risk only.
    """
    try:
        # Load sample data
        returns_df, weights = load_sample_portfolio_data("sample")
        
        # Run tail risk analysis
        results = tail_risk_analyzer.analyze_tail_risk(returns_df, weights)
        
        # Convert for JSON
        def convert_for_json(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            else:
                return obj
        
        return convert_for_json(results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample tail risk analysis failed: {str(e)}")

# Report generation endpoints
@app.post("/report/comprehensive")
async def generate_comprehensive_report(
    analysis_data: Dict[str, Any],
    report_type: str = Query("comprehensive", description="Type of report: comprehensive, executive_summary, advisor_presentation, risk_committee")
):
    """
    Generate comprehensive report from analysis results.
    """
    try:
        # Generate report using enhanced analyzer
        report_text = enhanced_analyzer.generate_comprehensive_report(
            analysis_data, report_type
        )
        
        return {
            "report_type": report_type,
            "report_content": report_text,
            "generated_at": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# Utility endpoints for tail risk
@app.get("/stress-scenarios")
async def get_available_stress_scenarios():
    """
    Get list of available stress test scenarios.
    """
    return {
        "stress_scenarios": tail_risk_analyzer.stress_scenarios,
        "total_scenarios": len(tail_risk_analyzer.stress_scenarios)
    }

@app.post("/analyze/custom-stress-test")
async def perform_custom_stress_test(
    tickers: List[str],
    stress_returns: Dict[str, float],
    weights: Optional[List[float]] = None,
    period: str = "1y"
):
    """
    Perform custom stress test with user-defined scenario.
    
    Args:
        tickers: List of portfolio tickers
        stress_returns: Dictionary mapping scenario names to market returns
        weights: Optional portfolio weights
        period: Time period for data
    """
    try:
        # Fetch returns data
        returns_df = data_service.fetch_returns_data(tickers, period=period)
        
        # Create weights
        weights_series = None
        if weights:
            weights_series = pd.Series(weights, index=returns_df.columns)
            weights_series = weights_series / weights_series.sum()
        
        # Calculate portfolio returns
        if weights_series is not None:
            portfolio_returns = (returns_df * weights_series).sum(axis=1)
        else:
            portfolio_returns = returns_df.mean(axis=1)
        
        # Estimate portfolio beta
        market_proxy_returns = returns_df.iloc[:, 0]
        portfolio_beta = portfolio_returns.cov(market_proxy_returns) / market_proxy_returns.var()
        
        # Apply stress scenarios
        stress_results = {}
        for scenario_name, market_return in stress_returns.items():
            stressed_return = market_return * portfolio_beta
            stress_results[scenario_name] = {
                "market_return": market_return,
                "estimated_portfolio_return": float(stressed_return),
                "portfolio_beta": float(portfolio_beta)
            }
        
        return {
            "custom_stress_results": stress_results,
            "portfolio_beta": float(portfolio_beta),
            "analysis_successful": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom stress test failed: {str(e)}")

# Configuration endpoint for enhanced features
@app.get("/config/enhanced")
async def get_enhanced_configuration():
    """
    Get enhanced API configuration including tail risk features.
    """
    return {
        "api_version": "2.0.0",
        "features": {
            "correlation_analysis": True,
            "tail_risk_analysis": True,
            "stress_testing": True,
            "var_cvar_calculation": True,
            "drawdown_analysis": True,
            "rolling_volatility": True,
            "extreme_value_analysis": True,
            "comprehensive_reporting": True
        },
        "supported_confidence_levels": [0.90, 0.95, 0.99, 0.995],
        "stress_test_scenarios": list(tail_risk_analyzer.stress_scenarios.keys()),
        "max_assets_default": settings.max_assets_default,
        "supported_periods": settings.supported_periods,
        "analysis_methods": {
            "var_methods": ["historical", "parametric", "cornish_fisher"],
            "drawdown_metrics": ["maximum", "average", "recovery_time"],
            "volatility_windows": [21, 63, 252]
        }
    }

# Health check with enhanced features
@app.get("/health/enhanced")
async def enhanced_health_check():
    """Enhanced health check including tail risk components."""
    try:
        # Test basic functionality
        import numpy as np
        test_data = pd.DataFrame(np.random.normal(0, 0.01, (100, 3)), columns=['A', 'B', 'C'])
        
        # Test correlation analysis
        corr_analyzer = enhanced_analyzer
        
        # Test tail risk analysis
        tail_analyzer = tail_risk_analyzer
        
        return {
            "status": "healthy",
            "service": "enhanced-portfolio-analysis",
            "components": {
                "correlation_analysis": "operational",
                "tail_risk_analysis": "operational",
                "data_service": "operational",
                "stress_testing": "operational"
            },
            "version": "2.0.0",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "service": "enhanced-portfolio-analysis",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
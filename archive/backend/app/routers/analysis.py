# app/routers/analysis.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import time
import uuid
import logging

from app.core.database import get_db
from app.models.portfolio import Portfolio, Analysis
from app.schemas.portfolio import AnalysisRequest, AnalysisResponse
from app.core.security import get_current_user
from app.models.portfolio import User
from app.services.data_service import DataService
from app.services.analysis_service import AnalysisService
from app.main import get_redis

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=dict)
async def run_portfolio_analysis(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """Run portfolio analysis"""
    
    # Get portfolio
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == request.portfolio_id,
            Portfolio.owner_id == current_user.id
        )
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Initialize services
    data_service = DataService(redis_client)
    analysis_service = AnalysisService(data_service)
    
    try:
        start_time = time.time()
        
        # Run analysis
        analysis_results = await analysis_service.analyze_portfolio(
            tickers=portfolio.tickers,
            weights=portfolio.weights,
            benchmark_ticker=portfolio.benchmark_ticker,
            start_date=request.start_date,
            analysis_types=request.analysis_types
        )
        
        computation_time = time.time() - start_time
        
        # Store results in database
        for analysis_type in request.analysis_types:
            if analysis_type in analysis_results:
                db_analysis = Analysis(
                    portfolio_id=portfolio.id,
                    analysis_type=analysis_type,
                    parameters=request.parameters or {},
                    results=analysis_results[analysis_type],
                    computation_time=computation_time
                )
                db.add(db_analysis)
        
        await db.commit()
        
        # Return complete results
        return {
            "portfolio_id": str(portfolio.id),
            "analysis_types": request.analysis_types,
            "computation_time": computation_time,
            "results": analysis_results
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/{portfolio_id}/history")
async def get_analysis_history(
    portfolio_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get analysis history for portfolio"""
    
    # Verify portfolio ownership
    portfolio_result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.owner_id == current_user.id
        )
    )
    portfolio = portfolio_result.scalar_one_or_none()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get analysis history
    result = await db.execute(
        select(Analysis).where(Analysis.portfolio_id == portfolio_id)
        .order_by(Analysis.created_at.desc())
        .limit(50)
    )
    analyses = result.scalars().all()
    
    return [
        {
            "id": str(analysis.id),
            "analysis_type": analysis.analysis_type,
            "created_at": analysis.created_at,
            "computation_time": analysis.computation_time,
            "results": analysis.results
        }
        for analysis in analyses
    ]

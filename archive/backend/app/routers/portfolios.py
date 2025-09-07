# app/routers/portfolios.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import uuid

from app.core.database import get_db
from app.models.portfolio import Portfolio
from app.schemas.portfolio import PortfolioCreate, PortfolioUpdate, PortfolioResponse
from app.core.security import get_current_user
from app.models.portfolio import User

router = APIRouter()

@router.post("/", response_model=PortfolioResponse)
async def create_portfolio(
    portfolio: PortfolioCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new portfolio"""
    db_portfolio = Portfolio(
        owner_id=current_user.id,
        name=portfolio.name,
        description=portfolio.description,
        tickers=portfolio.tickers,
        weights=portfolio.weights,
        benchmark_ticker=portfolio.benchmark_ticker
    )
    
    db.add(db_portfolio)
    await db.commit()
    await db.refresh(db_portfolio)
    
    return db_portfolio

@router.get("/", response_model=List[PortfolioResponse])
async def get_portfolios(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all portfolios for current user"""
    result = await db.execute(
        select(Portfolio).where(Portfolio.owner_id == current_user.id)
    )
    portfolios = result.scalars().all()
    return portfolios

@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific portfolio"""
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.owner_id == current_user.id
        )
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return portfolio

@router.put("/{portfolio_id}", response_model=PortfolioResponse)
async def update_portfolio(
    portfolio_id: uuid.UUID,
    portfolio_update: PortfolioUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update portfolio"""
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.owner_id == current_user.id
        )
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Update fields
    update_data = portfolio_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(portfolio, field, value)
    
    await db.commit()
    await db.refresh(portfolio)
    
    return portfolio

@router.delete("/{portfolio_id}")
async def delete_portfolio(
    portfolio_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete portfolio"""
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.owner_id == current_user.id
        )
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    await db.delete(portfolio)
    await db.commit()
    
    return {"message": "Portfolio deleted successfully"}

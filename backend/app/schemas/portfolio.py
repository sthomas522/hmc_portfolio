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

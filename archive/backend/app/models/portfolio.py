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

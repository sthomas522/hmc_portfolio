import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
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
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    if not data.empty and 'Close' in data.columns:
                        successful_data[ticker] = data['Close']
                    else:
                        failed_tickers.append(ticker)
                else:
                    # Multiple tickers
                    data = yf.download(batch, start=start_date, end=end_date, progress=False, auto_adjust=True)
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
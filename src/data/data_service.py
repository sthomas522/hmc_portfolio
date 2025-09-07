"""
Data service for fetching and processing portfolio holdings and returns data.

This module handles data retrieval from various sources and prepares it
for correlation analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta


class DataService:
    """
    Service for fetching and processing portfolio data.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_holdings_from_csv(self, file_path: str, 
                              ticker_column: str = 'Ticker',
                              weight_column: str = 'Portfolio Weight') -> pd.DataFrame:
        """
        Load holdings data from CSV file.
        
        Args:
            file_path: Path to CSV file
            ticker_column: Name of ticker column
            weight_column: Name of weight column
            
        Returns:
            DataFrame with cleaned holdings data
        """
        try:
            # Try different CSV reading strategies
            try:
                holdings = pd.read_csv(file_path)
            except pd.errors.ParserError:
                # Try with different parsing options
                holdings = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
            except UnicodeDecodeError:
                # Try different encoding
                holdings = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip')
            
            # Clean ticker symbols
            if ticker_column in holdings.columns:
                holdings[ticker_column] = holdings[ticker_column].astype(str)
                holdings[ticker_column] = holdings[ticker_column].str.strip()
                holdings[ticker_column] = holdings[ticker_column].str.replace('/', '-')
                
            # Clean weights if present
            if weight_column in holdings.columns:
                holdings[weight_column] = pd.to_numeric(holdings[weight_column], errors='coerce')
                holdings = holdings.dropna(subset=[weight_column])
                
            # Remove empty tickers
            holdings = holdings[holdings[ticker_column] != '']
            holdings = holdings[holdings[ticker_column] != 'nan']
            
            self.logger.info(f"Loaded {len(holdings)} holdings from {file_path}")
            return holdings
            
        except Exception as e:
            self.logger.error(f"Error loading holdings from {file_path}: {str(e)}")
            raise
    
    def fetch_returns_data(self, tickers: List[str], 
                          period: str = "1y",
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch returns data for given tickers.
        
        Args:
            tickers: List of ticker symbols
            period: Time period (e.g., "1y", "2y", "5y")
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            DataFrame with returns data
        """
        try:
            # Download price data
            if start_date and end_date:
                data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            else:
                data = yf.download(tickers, period=period, progress=False)
            
            if data.empty:
                raise ValueError("No data retrieved from yfinance")

            # Handle different yfinance return structures
            if len(tickers) == 1:
                # Single ticker - yfinance returns simple DataFrame
                if 'Adj Close' in data.columns:
                    prices = data[['Adj Close']].copy()
                    prices.columns = tickers
                elif 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = tickers
                else:
                    raise ValueError("No price data available")
            else:
                # Multiple tickers - check for MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Adj Close' in data.columns.get_level_values(0):
                        prices = data['Adj Close']
                    elif 'Close' in data.columns.get_level_values(0):
                        prices = data['Close']
                    else:
                        raise ValueError("No price data available")
                else:
                    # Fallback for simple columns
                    prices = data
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Remove columns with insufficient data
            min_observations = max(50, len(tickers) * 2)
            valid_columns = returns.count() >= min_observations
            returns = returns.loc[:, valid_columns]
            
            self.logger.info(f"Fetched returns for {len(returns.columns)} tickers over {len(returns)} days")
            return returns
            
        except Exception as e:
            self.logger.error(f"Error fetching returns data: {str(e)}")
            raise
    
    def fetch_portfolio_returns(self, holdings_df: pd.DataFrame,
                               ticker_column: str = 'Ticker',
                               weight_column: Optional[str] = None,
                               period: str = "1y",
                               max_assets: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fetch returns data for portfolio holdings.
        
        Args:
            holdings_df: DataFrame with holdings information
            ticker_column: Name of ticker column
            weight_column: Optional name of weight column
            period: Time period for returns
            max_assets: Optional limit on number of assets
            
        Returns:
            Tuple of (returns_df, weights_series)
        """
        # Get ticker list
        tickers = holdings_df[ticker_column].tolist()
        
        # Limit number of assets if specified
        if max_assets and len(tickers) > max_assets:
            if weight_column and weight_column in holdings_df.columns:
                # Keep top holdings by weight
                holdings_sorted = holdings_df.nlargest(max_assets, weight_column)
                tickers = holdings_sorted[ticker_column].tolist()
            else:
                # Keep first N tickers
                tickers = tickers[:max_assets]
        
        # Fetch returns data
        returns_df = self.fetch_returns_data(tickers, period=period)
        
        # Create weights series
        weights_series = None
        if weight_column and weight_column in holdings_df.columns:
            weights_dict = dict(zip(holdings_df[ticker_column], holdings_df[weight_column]))
            # Only include weights for tickers that have data
            valid_tickers = returns_df.columns.tolist()
            weights_series = pd.Series({ticker: weights_dict.get(ticker, 0) for ticker in valid_tickers})
            weights_series = weights_series / weights_series.sum()  # Normalize
        
        return returns_df, weights_series
    
    def validate_ticker_data(self, tickers: List[str], 
                           sample_size: int = 5) -> Dict[str, bool]:
        """
        Validate that tickers can be retrieved from data source.
        
        Args:
            tickers: List of ticker symbols to validate
            sample_size: Number of tickers to test
            
        Returns:
            Dictionary mapping ticker to validity
        """
        results = {}
        
        # Test a sample of tickers
        test_tickers = tickers[:sample_size] if len(tickers) > sample_size else tickers
        
        for ticker in test_tickers:
            try:
                data = yf.download(ticker, period="5d", progress=False)
                results[ticker] = not data.empty
            except:
                results[ticker] = False
        
        return results
    
    def clean_ticker_symbols(self, tickers: List[str]) -> List[str]:
        """
        Clean and standardize ticker symbols.
        
        Args:
            tickers: List of raw ticker symbols
            
        Returns:
            List of cleaned ticker symbols
        """
        cleaned = []
        
        for ticker in tickers:
            if pd.isna(ticker) or ticker == '':
                continue
                
            # Convert to string and strip whitespace
            ticker = str(ticker).strip().upper()
            
            # Common replacements
            ticker = ticker.replace('/', '-')  # BRK/B -> BRK-B
            ticker = ticker.replace('.', '-')  # BRK.B -> BRK-B
            
            # Remove invalid characters
            ticker = ''.join(c for c in ticker if c.isalnum() or c in ['-', '.'])
            
            if len(ticker) > 0:
                cleaned.append(ticker)
        
        return cleaned
    
    def get_data_quality_report(self, returns_df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate data quality report for returns data.
        
        Args:
            returns_df: DataFrame with returns data
            
        Returns:
            Dictionary with data quality metrics
        """
        report = {
            'num_assets': len(returns_df.columns),
            'num_observations': len(returns_df),
            'date_range': {
                'start': returns_df.index.min().strftime('%Y-%m-%d'),
                'end': returns_df.index.max().strftime('%Y-%m-%d')
            },
            'missing_data': {
                'total_missing': returns_df.isnull().sum().sum(),
                'assets_with_missing': (returns_df.isnull().sum() > 0).sum(),
                'missing_by_asset': returns_df.isnull().sum().to_dict()
            },
            'return_statistics': {
                'mean_return': returns_df.mean().mean(),
                'volatility': returns_df.std().mean(),
                'min_return': returns_df.min().min(),
                'max_return': returns_df.max().max()
            }
        }
        
        return report


# Utility functions for data processing
def prepare_returns_for_analysis(returns_df: pd.DataFrame,
                                min_observations: Optional[int] = None) -> pd.DataFrame:
    """
    Prepare returns data for correlation analysis.
    
    Args:
        returns_df: Raw returns DataFrame
        min_observations: Minimum required observations per asset
        
    Returns:
        Cleaned returns DataFrame
    """
    if min_observations is None:
        min_observations = max(50, returns_df.shape[1] * 2)
    
    # Remove assets with insufficient data
    valid_assets = returns_df.count() >= min_observations
    returns_clean = returns_df.loc[:, valid_assets]
    
    # Remove days with too much missing data
    max_missing_pct = 0.1  # Allow up to 10% missing data per day
    valid_days = returns_clean.isnull().sum(axis=1) / returns_clean.shape[1] <= max_missing_pct
    returns_clean = returns_clean.loc[valid_days]
    
    # Fill remaining missing values with 0 (conservative approach)
    returns_clean = returns_clean.fillna(0)
    
    return returns_clean


def load_sample_portfolio_data(portfolio_name: str = "sample") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load sample portfolio data for testing.
    
    Args:
        portfolio_name: Name of sample portfolio to load
        
    Returns:
        Tuple of (returns_df, weights_series)
    """
    if portfolio_name == "sample":
        # Create sample tech portfolio
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM']
        weights = [0.15, 0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.05, 0.05]
    else:
        raise ValueError(f"Unknown sample portfolio: {portfolio_name}")
    
    # Fetch data
    data_service = DataService()
    returns_df = data_service.fetch_returns_data(tickers, period="1y")
    
    # Create weights series for available tickers
    available_tickers = returns_df.columns.tolist()
    weights_dict = {ticker: weight for ticker, weight in zip(tickers, weights) if ticker in available_tickers}
    weights_series = pd.Series(weights_dict)
    weights_series = weights_series / weights_series.sum()  # Normalize
    
    return returns_df, weights_series
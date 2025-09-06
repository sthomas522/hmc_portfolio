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
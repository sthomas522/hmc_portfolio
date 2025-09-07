import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from scipy import stats
from scipy.linalg import LinAlgError
import warnings
from app.services.data_service import DataService

logger = logging.getLogger(__name__)

class ImprovedAnalysisService:
    """
    Enhanced portfolio analysis service with robust numerical methods
    """
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.correlation_regularization = 1e-6  # Small regularization factor
        self.max_condition_number = 1e12  # Maximum acceptable condition number
    
    async def analyze_portfolio(
        self,
        tickers: List[str],
        weights: Optional[List[float]] = None,
        benchmark_ticker: str = "SPY",
        start_date: str = None,
        analysis_types: List[str] = None
    ) -> Dict[str, Any]:
        """Enhanced portfolio analysis with improved data handling"""
        
        if analysis_types is None:
            analysis_types = ['basic', 'correlation']
        
        logger.info(f"Starting analysis for {len(tickers)} tickers")
        
        # Enhanced data fetching with diagnostics
        try:
            price_data, failed_tickers, metadata = await self.data_service.fetch_portfolio_data(
                tickers, start_date
            )
        except Exception as e:
            logger.error(f"Data fetching failed: {e}")
            raise ValueError(f"Could not fetch data for portfolio: {e}")
        
        # Data quality checks
        valid_tickers = price_data.columns.tolist()
        success_rate = len(valid_tickers) / len(tickers)
        
        logger.info(f"Data success rate: {len(valid_tickers)}/{len(tickers)} ({success_rate:.1%})")
        
        if success_rate < 0.5:
            logger.warning(f"Low data success rate: {success_rate:.1%}")
        
        if len(valid_tickers) < 3:
            raise ValueError(f"Insufficient data: only {len(valid_tickers)} tickers loaded successfully")
        
        # Adjust weights for successful data
        if weights:
            # Map weights to successful tickers
            original_tickers = [t for t in tickers if t not in failed_tickers]
            weight_mapping = dict(zip(original_tickers, weights))
            valid_weights = np.array([weight_mapping.get(t, 0) for t in valid_tickers])
            
            # Renormalize weights
            if valid_weights.sum() > 0:
                valid_weights = valid_weights / valid_weights.sum()
            else:
                valid_weights = np.array([1/len(valid_tickers)] * len(valid_tickers))
        else:
            valid_weights = np.array([1/len(valid_tickers)] * len(valid_tickers))
        
        # Enhanced return calculation with outlier handling
        returns = self._calculate_robust_returns(price_data)
        portfolio_returns = (returns * valid_weights).sum(axis=1)
        
        # Build comprehensive results
        results = {
            'metadata': {
                'portfolio_tickers': valid_tickers,
                'weights': valid_weights.tolist(),
                'failed_tickers': failed_tickers,
                'success_rate': success_rate,
                'benchmark_ticker': benchmark_ticker,
                'analysis_period': metadata['date_range'],
                'total_observations': len(returns),
                'data_quality': self._assess_data_quality(returns)
            }
        }
        
        if 'basic' in analysis_types:
            results['basic'] = self._basic_analysis(portfolio_returns)
        
        if 'correlation' in analysis_types:
            results['correlation'] = self._robust_correlation_analysis(returns, valid_weights)
        
        return results
    
    def _calculate_robust_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns with outlier handling"""
        
        # Basic returns
        returns = price_data.pct_change().dropna()
        
        # Identify and handle extreme outliers (>5 standard deviations)
        for col in returns.columns:
            series = returns[col]
            mean_ret = series.mean()
            std_ret = series.std()
            
            # Cap extreme outliers at 5 standard deviations
            outlier_threshold = 5 * std_ret
            returns[col] = series.clip(
                lower=mean_ret - outlier_threshold,
                upper=mean_ret + outlier_threshold
            )
        
        # Remove rows with too many NaN values
        returns = returns.dropna(thresh=len(returns.columns) * 0.8)
        
        logger.info(f"Returns calculated: {len(returns)} observations across {len(returns.columns)} assets")
        
        return returns
    
    def _assess_data_quality(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics"""
        
        nan_counts = returns.isnull().sum()
        zero_counts = (returns == 0).sum()
        
        return {
            'nan_percentage': (nan_counts.sum() / (len(returns) * len(returns.columns))),
            'zero_percentage': (zero_counts.sum() / (len(returns) * len(returns.columns))),
            'outlier_percentage': self._count_outliers(returns),
            'min_observations': len(returns),
            'assets_with_issues': nan_counts[nan_counts > len(returns) * 0.1].index.tolist()
        }
    
    def _count_outliers(self, returns: pd.DataFrame) -> float:
        """Count percentage of potential outliers"""
        outliers = 0
        total = 0
        
        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) > 10:
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                outlier_mask = (series < q1 - 3*iqr) | (series > q3 + 3*iqr)
                outliers += outlier_mask.sum()
                total += len(series)
        
        return outliers / total if total > 0 else 0
    
    def _robust_correlation_analysis(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """Enhanced correlation analysis with numerical stability"""
        
        logger.info(f"Starting robust correlation analysis for {len(returns.columns)} assets")
        
        # Calculate correlation matrix with robust methods
        correlation_matrix = self._calculate_robust_correlation(returns)
        
        # Numerical stability checks
        stability_results = self._check_numerical_stability(correlation_matrix)
        
        # Calculate eigenvalues with enhanced error handling
        eigenvalue_results = self._robust_eigenvalue_analysis(correlation_matrix)
        
        # Portfolio-specific metrics
        portfolio_metrics = self._calculate_portfolio_metrics(correlation_matrix, weights)
        
        # Correlation statistics
        corr_stats = self._calculate_correlation_statistics(correlation_matrix)
        
        # Combine all results
        results = {
            **eigenvalue_results,
            **portfolio_metrics,
            **corr_stats,
            'numerical_stability': stability_results,
            'analysis_warnings': self._generate_analysis_warnings(stability_results, eigenvalue_results)
        }
        
        return results
    
    def _calculate_robust_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix with regularization"""
        
        # Use Pearson correlation with pairwise complete observations
        corr_matrix = returns.corr(method='pearson', min_periods=30)
        
        # Fill any remaining NaN values with 0 (uncorrelated)
        corr_matrix = corr_matrix.fillna(0)
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(corr_matrix.values, 1.0)
        
        # Apply small regularization to improve numerical stability
        n = len(corr_matrix)
        regularized_matrix = corr_matrix + self.correlation_regularization * np.eye(n)
        
        # Renormalize diagonal
        np.fill_diagonal(regularized_matrix.values, 1.0)
        
        return regularized_matrix
    
    def _check_numerical_stability(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive numerical stability assessment"""
        
        matrix = correlation_matrix.values
        
        # Check if matrix is symmetric
        is_symmetric = np.allclose(matrix, matrix.T, rtol=1e-10)
        
        # Check if matrix is positive semidefinite
        eigenvals_check = np.linalg.eigvals(matrix)
        min_eigenval = np.min(np.real(eigenvals_check))
        is_positive_semidefinite = min_eigenval >= -1e-8
        
        # Calculate condition number
        try:
            condition_number = np.linalg.cond(matrix)
        except LinAlgError:
            condition_number = np.inf
        
        # Check for numerical issues
        has_nan = np.isnan(matrix).any()
        has_inf = np.isinf(matrix).any()
        
        # Assess matrix rank
        rank = np.linalg.matrix_rank(matrix, tol=1e-12)
        expected_rank = matrix.shape[0]
        
        return {
            'is_symmetric': is_symmetric,
            'is_positive_semidefinite': is_positive_semidefinite,
            'min_eigenvalue': float(min_eigenval),
            'condition_number': float(condition_number),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'matrix_rank': rank,
            'expected_rank': expected_rank,
            'rank_deficient': rank < expected_rank,
            'well_conditioned': condition_number < self.max_condition_number
        }
    
    def _robust_eigenvalue_analysis(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Robust eigenvalue computation with error handling"""
        
        matrix = correlation_matrix.values
        
        try:
            # Use scipy's more robust eigenvalue computation
            eigenvalues = np.linalg.eigvals(matrix)
            
            # Handle complex eigenvalues
            if np.any(np.iscomplex(eigenvalues)):
                logger.warning(f"Complex eigenvalues detected: {np.sum(np.iscomplex(eigenvalues))} out of {len(eigenvalues)}")
                eigenvalues = np.real(eigenvalues)
            
            # Remove numerical zeros and negative values
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            # Calculate effective rank (participation ratio)
            eigenvalue_weights = eigenvalues / eigenvalues.sum()
            effective_rank = 1 / np.sum(eigenvalue_weights**2)
            
            # Calculate concentration metrics
            num_assets = len(correlation_matrix)
            concentration_ratio = effective_rank / num_assets
            
            # Explained variance analysis
            explained_variance_ratios = eigenvalue_weights
            cumulative_variance = np.cumsum(explained_variance_ratios)
            
            return {
                'effective_rank': float(effective_rank),
                'concentration_ratio': float(concentration_ratio),
                'eigenvalues': eigenvalues.tolist()[:20],  # Top 20 eigenvalues
                'explained_variance_ratios': explained_variance_ratios.tolist()[:20],
                'cumulative_variance_explained': cumulative_variance.tolist()[:20],
                'num_positive_eigenvalues': len(eigenvalues),
                'eigenvalue_analysis_successful': True
            }
            
        except Exception as e:
            logger.error(f"Eigenvalue analysis failed: {e}")
            
            # Fallback to simple analysis
            num_assets = len(correlation_matrix)
            return {
                'effective_rank': float(num_assets * 0.5),  # Conservative estimate
                'concentration_ratio': 0.5,
                'eigenvalues': [],
                'explained_variance_ratios': [],
                'cumulative_variance_explained': [],
                'num_positive_eigenvalues': 0,
                'eigenvalue_analysis_successful': False,
                'eigenvalue_error': str(e)
            }
    
    def _calculate_portfolio_metrics(self, correlation_matrix: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
        """Calculate portfolio-specific correlation metrics"""
        
        matrix = correlation_matrix.values
        
        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(matrix, weights))
        
        # Average correlation calculation
        if len(weights) > 1:
            # Weighted average correlation
            weighted_corr_sum = 0
            weight_sum = 0
            
            for i in range(len(weights)):
                for j in range(i + 1, len(weights)):
                    corr_ij = matrix[i, j]
                    weight_ij = weights[i] * weights[j]
                    weighted_corr_sum += corr_ij * weight_ij
                    weight_sum += weight_ij
            
            average_correlation = weighted_corr_sum / weight_sum if weight_sum > 0 else 0
        else:
            average_correlation = 0
        
        return {
            'portfolio_concentration': {
                'portfolio_variance': float(portfolio_variance),
                'average_correlation': float(average_correlation),
                'diversification_ratio': float(1 / np.sqrt(portfolio_variance)) if portfolio_variance > 0 else 0
            }
        }
    
    def _calculate_correlation_statistics(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive correlation statistics"""
        
        matrix = correlation_matrix.values
        
        # Get upper triangular correlations (excluding diagonal)
        upper_tri_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        correlations = matrix[upper_tri_mask]
        
        return {
            'correlation_statistics': {
                'mean_correlation': float(np.mean(correlations)),
                'median_correlation': float(np.median(correlations)),
                'std_correlation': float(np.std(correlations)),
                'max_correlation': float(np.max(correlations)),
                'min_correlation': float(np.min(correlations)),
                'correlations_above_0_5': int(np.sum(correlations > 0.5)),
                'correlations_above_0_7': int(np.sum(correlations > 0.7)),
                'correlations_above_0_9': int(np.sum(correlations > 0.9)),
                'total_correlations': len(correlations)
            }
        }
    
    def _generate_analysis_warnings(self, stability: Dict, eigenvalue_results: Dict) -> List[str]:
        """Generate warnings based on analysis results"""
        
        warnings = []
        
        if not stability['well_conditioned']:
            warnings.append(f"Matrix is ill-conditioned (condition number: {stability['condition_number']:.2e})")
        
        if not stability['is_positive_semidefinite']:
            warnings.append(f"Correlation matrix is not positive semidefinite (min eigenvalue: {stability['min_eigenvalue']:.6f})")
        
        if stability['rank_deficient']:
            warnings.append(f"Matrix is rank deficient ({stability['matrix_rank']}/{stability['expected_rank']})")
        
        if not eigenvalue_results['eigenvalue_analysis_successful']:
            warnings.append("Eigenvalue analysis failed - results may be unreliable")
        
        if eigenvalue_results.get('concentration_ratio', 0) < 0.1:
            warnings.append("Extremely high correlation detected - portfolio may lack diversification")
        
        return warnings
    
    def _basic_analysis(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """Enhanced basic portfolio analysis"""
        
        # Annualized metrics (assuming daily returns)
        trading_days = 252
        annual_return = portfolio_returns.mean() * trading_days
        annual_vol = portfolio_returns.std() * np.sqrt(trading_days)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Risk metrics
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Drawdown analysis
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Additional statistics
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        
        return {
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'total_return': float((1 + portfolio_returns).prod() - 1),
            'win_rate': float((portfolio_returns > 0).mean()),
            'best_day': float(portfolio_returns.max()),
            'worst_day': float(portfolio_returns.min())
        }
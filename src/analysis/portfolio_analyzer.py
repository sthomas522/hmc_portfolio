"""
Complete portfolio analyzer with correlation, tail risk, and factor attribution.

This module provides comprehensive portfolio risk assessment including:
- Correlation structure analysis with eigenvalue decomposition
- Tail risk metrics (VaR, CVaR, drawdowns, stress testing)
- Factor attribution (economic interpretation of eigenvalue factors)
- Integrated insights across all analyses
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from analysis.mp_analysis import MarchenkoPosturAnalyzer
from analysis.correlation_analysis import (
    calculate_effective_rank_metrics,
    analyze_risk_concentration,
    clean_correlation_matrix,
    validate_correlation_data,
    perform_eigenvalue_decomposition,
    generate_correlation_interpretation,
    generate_advisor_talking_points
)
from data.data_service import DataService


@dataclass
class FactorInterpretation:
    """Structure for factor interpretation results."""
    factor_name: str
    factor_type: str  # 'market', 'sector', 'style', 'idiosyncratic'
    confidence: float  # 0-1 confidence in interpretation
    description: str
    top_loadings: List[Tuple[str, float]]  # (ticker, loading) pairs
    correlation_with_benchmark: float
    variance_explained: float


class TailRiskAnalyzer:
    """
    Comprehensive tail risk analysis for portfolio returns.
    
    Provides Value at Risk, Conditional VaR, maximum drawdown analysis,
    stress testing against historical events, and rolling risk metrics.
    """
    
    def __init__(self):
        # Historical stress test scenarios (returns for major market events)
        self.stress_scenarios = {
            '2008_financial_crisis': {
                'name': '2008 Financial Crisis',
                'period': '2008-09-15 to 2009-03-09',
                'market_return': -0.567,  # S&P 500 peak to trough
                'description': 'Lehman Brothers collapse and financial system stress'
            },
            '2020_covid_crash': {
                'name': 'COVID-19 Market Crash',
                'period': '2020-02-19 to 2020-03-23',
                'market_return': -0.338,  # S&P 500 peak to trough
                'description': 'Pandemic-induced market sell-off'
            },
            '2018_volatility_spike': {
                'name': '2018 Volatility Spike',
                'period': '2018-01-26 to 2018-02-09',
                'market_return': -0.108,  # S&P 500 correction
                'description': 'VIX spike and algorithmic selling'
            },
            '2022_rate_shock': {
                'name': '2022 Rate Shock',
                'period': '2022-01-03 to 2022-06-17',
                'market_return': -0.238,  # S&P 500 bear market start
                'description': 'Federal Reserve aggressive rate hiking cycle'
            }
        }
    
    def analyze_tail_risk(self, returns_data: pd.DataFrame, 
                         weights: Optional[pd.Series] = None,
                         confidence_levels: List[float] = [0.95, 0.99, 0.995]) -> Dict[str, Any]:
        """
        Perform comprehensive tail risk analysis on portfolio returns.
        
        Args:
            returns_data: DataFrame with asset returns
            weights: Optional portfolio weights
            confidence_levels: Confidence levels for VaR/CVaR calculation
            
        Returns:
            Dictionary containing all tail risk metrics
        """
        try:
            # Calculate portfolio returns
            if weights is not None:
                portfolio_returns = (returns_data * weights).sum(axis=1)
            else:
                # Equal weights if not provided
                portfolio_returns = returns_data.mean(axis=1)
            
            results = {
                'analysis_successful': True,
                'portfolio_summary': self._calculate_return_summary(portfolio_returns),
                'var_cvar_analysis': self._calculate_var_cvar(portfolio_returns, confidence_levels),
                'drawdown_analysis': self._calculate_drawdown_metrics(portfolio_returns),
                'stress_testing': self._perform_stress_testing(returns_data, weights),
                'rolling_volatility': self._calculate_rolling_volatility(portfolio_returns),
                'tail_dependency': self._analyze_tail_dependency(returns_data),
                'extreme_value_analysis': self._perform_extreme_value_analysis(portfolio_returns),
                'risk_interpretation': []
            }
            
            # Generate comprehensive interpretation
            results['risk_interpretation'] = self._generate_risk_interpretation(results)
            results['advisor_risk_talking_points'] = self._generate_advisor_risk_points(results)
            
            return results
        except Exception as e:
            return {
                'analysis_successful': False,
                'error': str(e)
            }
    
    def _calculate_return_summary(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """Calculate basic return and volatility statistics."""
        return {
            'total_observations': len(portfolio_returns),
            'annualized_return': float(portfolio_returns.mean() * 252),
            'annualized_volatility': float(portfolio_returns.std() * np.sqrt(252)),
            'sharpe_ratio': float((portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))),
            'skewness': float(stats.skew(portfolio_returns)),
            'excess_kurtosis': float(stats.kurtosis(portfolio_returns)),
            'jarque_bera_test': float(stats.jarque_bera(portfolio_returns)[1])  # p-value
        }
    
    def _calculate_var_cvar(self, portfolio_returns: pd.Series, 
                           confidence_levels: List[float]) -> Dict[str, Any]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        var_cvar_results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            
            # Historical VaR (empirical quantile)
            historical_var = np.percentile(portfolio_returns, alpha * 100)
            
            # Historical CVaR (expected shortfall)
            tail_losses = portfolio_returns[portfolio_returns <= historical_var]
            historical_cvar = tail_losses.mean() if len(tail_losses) > 0 else historical_var
            
            # Parametric VaR (assuming normal distribution)
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            parametric_var = stats.norm.ppf(alpha, mean_return, std_return)
            
            var_cvar_results[f'{confidence:.1%}'] = {
                'historical_var_daily': float(historical_var),
                'historical_cvar_daily': float(historical_cvar),
                'parametric_var_daily': float(parametric_var),
                'historical_var_annual': float(historical_var * np.sqrt(252)),
                'historical_cvar_annual': float(historical_cvar * np.sqrt(252)),
                'tail_observations': len(tail_losses)
            }
        
        return var_cvar_results
    
    def _calculate_drawdown_metrics(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """Calculate maximum drawdown and related metrics."""
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdowns.min()
        max_dd_date = drawdowns.idxmin()
        
        # Find peak before max drawdown
        max_dd_peak_date = running_max.loc[:max_dd_date].idxmax()
        
        # Recovery analysis
        recovery_date = None
        if max_dd_date < cumulative_returns.index[-1]:
            peak_value = running_max.loc[max_dd_date]
            post_trough = cumulative_returns.loc[max_dd_date:]
            recovery_mask = post_trough >= peak_value
            if recovery_mask.any():
                recovery_date = post_trough[recovery_mask].index[0]
        
        # Calculate drawdown duration
        dd_duration_days = (max_dd_date - max_dd_peak_date).days if max_dd_peak_date else None
        recovery_duration_days = (recovery_date - max_dd_date).days if recovery_date else None
        
        # Additional drawdown statistics
        negative_periods = drawdowns[drawdowns < 0]
        avg_drawdown = negative_periods.mean() if len(negative_periods) > 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        annual_return = portfolio_returns.mean() * 252
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'maximum_drawdown': float(max_drawdown),
            'max_drawdown_date': max_dd_date.strftime('%Y-%m-%d') if max_dd_date else None,
            'peak_date': max_dd_peak_date.strftime('%Y-%m-%d') if max_dd_peak_date else None,
            'recovery_date': recovery_date.strftime('%Y-%m-%d') if recovery_date else None,
            'drawdown_duration_days': dd_duration_days,
            'recovery_duration_days': recovery_duration_days,
            'average_drawdown': float(avg_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'periods_in_drawdown': len(negative_periods),
            'total_periods': len(drawdowns)
        }
    
    def _perform_stress_testing(self, returns_data: pd.DataFrame, 
                               weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Perform stress testing against historical market scenarios."""
        stress_results = {}
        
        # Calculate portfolio returns
        if weights is not None:
            portfolio_returns = (returns_data * weights).sum(axis=1)
        else:
            portfolio_returns = returns_data.mean(axis=1)
        
        # Estimate portfolio beta (simplified - assume first asset represents market)
        market_proxy_returns = returns_data.iloc[:, 0]
        portfolio_beta = portfolio_returns.cov(market_proxy_returns) / market_proxy_returns.var()
        
        for scenario_id, scenario in self.stress_scenarios.items():
            # Apply scenario stress
            stressed_return = scenario['market_return'] * portfolio_beta
            
            stress_results[scenario_id] = {
                'scenario_name': scenario['name'],
                'scenario_period': scenario['period'],
                'market_return': scenario['market_return'],
                'estimated_portfolio_return': float(stressed_return),
                'portfolio_value_impact': float(stressed_return),
                'description': scenario['description']
            }
        
        return stress_results
    
    def _calculate_rolling_volatility(self, portfolio_returns: pd.Series, 
                                    windows: List[int] = [21, 63, 252]) -> Dict[str, Any]:
        """Calculate rolling volatility metrics to identify regime changes."""
        rolling_vol_results = {}
        
        for window in windows:
            if len(portfolio_returns) >= window:
                rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
                
                rolling_vol_results[f'{window}_day'] = {
                    'window_days': window,
                    'current_volatility': float(rolling_vol.iloc[-1]) if not rolling_vol.empty else None,
                    'average_volatility': float(rolling_vol.mean()),
                    'max_volatility': float(rolling_vol.max()),
                    'min_volatility': float(rolling_vol.min()),
                    'volatility_percentile_current': float(rolling_vol.rank(pct=True).iloc[-1] * 100) if not rolling_vol.empty else None
                }
        
        return rolling_vol_results
    
    def _analyze_tail_dependency(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze tail dependency between assets."""
        n_assets = returns_data.shape[1]
        if n_assets < 2:
            return {'tail_dependency_available': False}
        
        # Calculate lower tail dependency (correlation in bottom 10% of returns)
        lower_threshold = returns_data.quantile(0.1)
        upper_threshold = returns_data.quantile(0.9)
        
        # Count simultaneous extreme events
        simultaneous_lower_tail = ((returns_data <= lower_threshold).sum(axis=1) >= 2).sum()
        simultaneous_upper_tail = ((returns_data >= upper_threshold).sum(axis=1) >= 2).sum()
        
        total_observations = len(returns_data)
        
        return {
            'tail_dependency_available': True,
            'simultaneous_lower_tail_events': int(simultaneous_lower_tail),
            'simultaneous_upper_tail_events': int(simultaneous_upper_tail),
            'lower_tail_dependency_ratio': float(simultaneous_lower_tail / total_observations),
            'upper_tail_dependency_ratio': float(simultaneous_upper_tail / total_observations),
            'total_observations': total_observations
        }
    
    def _perform_extreme_value_analysis(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """Perform extreme value analysis using block maxima method."""
        # Split returns into blocks (monthly blocks)
        returns_monthly = portfolio_returns.resample('M').min()  # Monthly minimums (worst returns)
        
        if len(returns_monthly) < 12:
            return {'extreme_value_analysis_available': False}
        
        # Fit Generalized Extreme Value distribution
        try:
            # Fit GEV to monthly minimums (block minima for losses)
            gev_params = stats.genextreme.fit(-returns_monthly)  # Fit to negative returns (losses)
            
            # Estimate extreme quantiles
            extreme_1_year = -stats.genextreme.ppf(0.99, *gev_params)  # 1% annual probability
            extreme_10_year = -stats.genextreme.ppf(0.999, *gev_params)  # 0.1% annual probability
            
            return {
                'extreme_value_analysis_available': True,
                'gev_fitted': True,
                'estimated_1_year_worst_case': float(extreme_1_year),
                'estimated_10_year_worst_case': float(extreme_10_year),
                'monthly_observations': len(returns_monthly)
            }
        except:
            return {
                'extreme_value_analysis_available': True,
                'gev_fitted': False,
                'error': 'Could not fit GEV distribution'
            }
    
    def _generate_risk_interpretation(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive risk interpretation."""
        interpretation = []
        
        # VaR/CVaR insights
        var_99 = results['var_cvar_analysis']['99.0%']['historical_var_daily']
        cvar_99 = results['var_cvar_analysis']['99.0%']['historical_cvar_daily']
        
        interpretation.append(f"Tail Risk Assessment:")
        interpretation.append(f"• 99% VaR: {var_99:.2%} daily loss expected 1% of the time")
        interpretation.append(f"• 99% CVaR: {cvar_99:.2%} average loss when exceeding VaR")
        
        # Drawdown insights
        max_dd = results['drawdown_analysis']['maximum_drawdown']
        dd_duration = results['drawdown_analysis']['drawdown_duration_days']
        recovery_duration = results['drawdown_analysis']['recovery_duration_days']
        
        interpretation.append(f"Drawdown Analysis:")
        interpretation.append(f"• Maximum drawdown: {max_dd:.1%}")
        if dd_duration:
            interpretation.append(f"• Drawdown lasted {dd_duration} days")
        if recovery_duration:
            interpretation.append(f"• Recovery took {recovery_duration} days")
        elif recovery_duration is None:
            interpretation.append(f"• Portfolio has not fully recovered from maximum drawdown")
        
        return interpretation
    
    def _generate_advisor_risk_points(self, results: Dict[str, Any]) -> List[str]:
        """Generate advisor-ready risk talking points."""
        talking_points = []
        
        # Key risk headline
        var_95 = abs(results['var_cvar_analysis']['95.0%']['historical_var_daily'])
        max_dd = abs(results['drawdown_analysis']['maximum_drawdown'])
        
        talking_points.append(f"Risk Profile: 95% confident daily losses won't exceed {var_95:.1%}")
        talking_points.append(f"Historical Worst Case: {max_dd:.1%} maximum drawdown experienced")
        
        # Stress test summary
        stress_losses = [abs(s['estimated_portfolio_return']) for s in results['stress_testing'].values()]
        avg_stress_loss = np.mean(stress_losses)
        talking_points.append(f"Crisis Resilience: Average {avg_stress_loss:.1%} loss in major market stress scenarios")
        
        return talking_points


class FactorAttributionAnalyzer:
    """
    Analyzes portfolio factors to provide economic interpretations.
    
    Maps eigenvalue factors from correlation analysis to:
    - Market factors (broad market exposure)
    - Sector factors (industry concentration)
    - Style factors (growth/value, size, momentum)
    - Idiosyncratic factors (company-specific)
    """
    
    def __init__(self):
        # Sector mapping for individual stocks
        self.sector_keywords = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'V', 'MA'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'LLY', 'ABT', 'DHR', 'BMY'],
            'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'TGT'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE'],
            'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'UNP']
        }
    
    def analyze_factor_attribution(self, returns_data: pd.DataFrame,
                                  eigenvalues: np.ndarray,
                                  eigenvectors: np.ndarray,
                                  num_factors: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive factor attribution analysis.
        
        Args:
            returns_data: DataFrame with asset returns
            eigenvalues: Eigenvalues from correlation matrix decomposition
            eigenvectors: Eigenvectors from correlation matrix decomposition
            num_factors: Number of top factors to analyze
            
        Returns:
            Dictionary containing factor attribution results
        """
        try:
            # Select top factors by eigenvalue magnitude
            num_factors = min(num_factors, len(eigenvalues))
            top_factor_indices = np.argsort(eigenvalues)[::-1][:num_factors]
            
            factor_interpretations = []
            
            for i, factor_idx in enumerate(top_factor_indices):
                eigenvalue = eigenvalues[factor_idx]
                eigenvector = eigenvectors[:, factor_idx]
                
                # Analyze this factor
                interpretation = self._interpret_single_factor(
                    returns_data, eigenvector, eigenvalue, i+1
                )
                factor_interpretations.append(interpretation)
            
            # Generate sector exposure analysis
            sector_exposure = self._analyze_sector_exposure(
                returns_data.columns, eigenvectors, top_factor_indices
            )
            
            return {
                'attribution_successful': True,
                'num_factors_analyzed': num_factors,
                'factor_interpretations': [self._factor_to_dict(f) for f in factor_interpretations],
                'sector_exposure': sector_exposure,
                'attribution_summary': self._generate_attribution_summary(factor_interpretations),
                'economic_insights': self._generate_economic_insights(factor_interpretations, sector_exposure)
            }
            
        except Exception as e:
            return {
                'attribution_successful': False,
                'error': str(e),
                'fallback_reason': 'Factor attribution analysis failed'
            }
    
    def _interpret_single_factor(self, returns_data: pd.DataFrame,
                                eigenvector: np.ndarray, 
                                eigenvalue: float,
                                factor_number: int) -> FactorInterpretation:
        """Interpret a single factor by analyzing its loadings and correlations."""
        tickers = returns_data.columns.tolist()
        loadings = eigenvector
        
        # Get top positive and negative loadings
        top_positive_idx = np.argsort(loadings)[-3:][::-1]
        top_negative_idx = np.argsort(loadings)[:3]
        
        top_loadings = []
        for idx in top_positive_idx:
            if abs(loadings[idx]) > 0.1:  # Only significant loadings
                top_loadings.append((tickers[idx], float(loadings[idx])))
        
        for idx in top_negative_idx:
            if abs(loadings[idx]) > 0.1:
                top_loadings.append((tickers[idx], float(loadings[idx])))
        
        # Determine factor type and interpretation
        factor_type, factor_name, confidence, description = self._classify_factor(
            tickers, loadings, factor_number
        )
        
        # Calculate variance explained
        variance_explained = float(eigenvalue / len(tickers))
        
        # Try to correlate with market benchmark (simplified)
        benchmark_correlation = self._estimate_market_correlation(loadings)
        
        return FactorInterpretation(
            factor_name=factor_name,
            factor_type=factor_type,
            confidence=confidence,
            description=description,
            top_loadings=top_loadings,
            correlation_with_benchmark=benchmark_correlation,
            variance_explained=variance_explained
        )
    
    def _classify_factor(self, tickers: List[str], loadings: np.ndarray, 
                        factor_number: int) -> Tuple[str, str, float, str]:
        """Classify factor based on loading patterns and ticker analysis."""
        # Analyze loading concentration
        loading_concentration = np.sum(loadings**2)
        max_loading = np.max(np.abs(loadings))
        
        # Get dominant tickers
        significant_mask = np.abs(loadings) > 0.3
        dominant_tickers = [tickers[i] for i in range(len(tickers)) if significant_mask[i]]
        
        # Sector analysis
        sector_concentration = self._analyze_sector_concentration(dominant_tickers)
        
        # Classification logic
        if factor_number == 1 and loading_concentration > 0.8:
            # First factor with high concentration - likely market factor
            return 'market', 'Market Factor', 0.9, 'Broad market exposure driving portfolio correlation'
        
        elif len(dominant_tickers) <= 2 and max_loading > 0.6:
            # Single stock dominance
            main_ticker = tickers[np.argmax(np.abs(loadings))]
            return 'idiosyncratic', f'{main_ticker} Specific Factor', 0.8, f'Factor driven primarily by {main_ticker}'
        
        elif sector_concentration['max_sector_weight'] > 0.7:
            # Sector concentration
            main_sector = sector_concentration['dominant_sector']
            confidence = 0.7 + 0.2 * sector_concentration['max_sector_weight']
            return 'sector', f'{main_sector} Sector Factor', confidence, f'Factor representing {main_sector} sector exposure'
        
        else:
            # Mixed factor
            return 'mixed', f'Mixed Factor {factor_number}', 0.5, 'Factor with mixed sector and style characteristics'
    
    def _analyze_sector_concentration(self, tickers: List[str]) -> Dict[str, Any]:
        """Analyze sector concentration in a list of tickers."""
        sector_counts = {}
        total_tickers = len(tickers)
        
        for ticker in tickers:
            sector_found = False
            for sector, sector_tickers in self.sector_keywords.items():
                if ticker in sector_tickers:
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    sector_found = True
                    break
            
            if not sector_found:
                sector_counts['Other'] = sector_counts.get('Other', 0) + 1
        
        if not sector_counts:
            return {'dominant_sector': 'Mixed', 'max_sector_weight': 0, 'sector_distribution': {}}
        
        dominant_sector = max(sector_counts, key=sector_counts.get)
        max_count = sector_counts[dominant_sector]
        max_weight = max_count / total_tickers if total_tickers > 0 else 0
        
        return {
            'dominant_sector': dominant_sector,
            'max_sector_weight': max_weight,
            'sector_distribution': sector_counts
        }
    
    def _estimate_market_correlation(self, loadings: np.ndarray) -> float:
        """Estimate correlation with market factor based on loading uniformity."""
        # Market factors typically have positive, relatively uniform loadings
        mean_loading = np.mean(loadings)
        loading_std = np.std(loadings)
        positive_ratio = np.sum(loadings > 0) / len(loadings)
        
        # High positive mean, low std, high positive ratio = market-like
        market_score = (mean_loading * positive_ratio) / (1 + loading_std)
        
        # Normalize to [-1, 1]
        return float(np.tanh(market_score * 2))
    
    def _analyze_sector_exposure(self, tickers: List[str],
                                eigenvectors: np.ndarray,
                                factor_indices: np.ndarray) -> Dict[str, Any]:
        """Analyze sector exposure for each factor."""
        sector_exposure = {}
        
        # Create sector mapping for tickers
        ticker_sectors = {}
        for ticker in tickers:
            ticker_sectors[ticker] = 'Other'
            for sector, sector_tickers in self.sector_keywords.items():
                if ticker in sector_tickers:
                    ticker_sectors[ticker] = sector
                    break
        
        # Calculate sector loadings for each factor
        for i, factor_idx in enumerate(factor_indices):
            factor_loadings = eigenvectors[:, factor_idx]
            factor_name = f'Factor_{i+1}'
            
            sector_loadings = {}
            for j, ticker in enumerate(tickers):
                sector = ticker_sectors[ticker]
                if sector not in sector_loadings:
                    sector_loadings[sector] = []
                sector_loadings[sector].append(factor_loadings[j])
            
            # Calculate average loading by sector
            sector_avg_loadings = {}
            for sector, loadings in sector_loadings.items():
                sector_avg_loadings[sector] = {
                    'average_loading': float(np.mean(loadings)),
                    'total_exposure': float(np.sum(np.abs(loadings))),
                    'num_stocks': len(loadings)
                }
            
            sector_exposure[factor_name] = sector_avg_loadings
        
        return sector_exposure
    
    def _generate_attribution_summary(self, interpretations: List[FactorInterpretation]) -> List[str]:
        """Generate human-readable summary of factor attribution."""
        summary = []
        
        # Overall factor composition
        factor_types = [f.factor_type for f in interpretations]
        type_counts = {t: factor_types.count(t) for t in set(factor_types)}
        
        summary.append(f"Factor Composition: {len(interpretations)} factors analyzed")
        
        if 'market' in type_counts:
            summary.append(f"• {type_counts['market']} market factor(s) - broad market exposure")
        if 'sector' in type_counts:
            summary.append(f"• {type_counts['sector']} sector factor(s) - industry concentration")
        if 'idiosyncratic' in type_counts:
            summary.append(f"• {type_counts['idiosyncratic']} stock-specific factor(s)")
        
        # Factor concentration
        total_variance = sum(f.variance_explained for f in interpretations)
        summary.append(f"Top factors explain {total_variance:.1%} of portfolio variance")
        
        return summary
    
    def _generate_economic_insights(self, interpretations: List[FactorInterpretation],
                                   sector_exposure: Dict[str, Any]) -> List[str]:
        """Generate actionable economic insights from factor analysis."""
        insights = []
        
        # Market factor insights
        market_factors = [f for f in interpretations if f.factor_type == 'market']
        if market_factors:
            market_weight = sum(f.variance_explained for f in market_factors)
            if market_weight > 0.5:
                insights.append(f"High market exposure ({market_weight:.1%}) - portfolio vulnerable to systematic market moves")
            else:
                insights.append(f"Moderate market exposure ({market_weight:.1%}) - some protection from market volatility")
        
        # Sector concentration insights
        sector_factors = [f for f in interpretations if f.factor_type == 'sector']
        if len(sector_factors) >= 2:
            insights.append("Multiple sector factors detected - portfolio has sector concentration risk")
        elif len(sector_factors) == 1:
            sector_name = sector_factors[0].factor_name
            insights.append(f"Single sector concentration in {sector_name}")
        
        return insights
    
    def _factor_to_dict(self, factor: FactorInterpretation) -> Dict[str, Any]:
        """Convert FactorInterpretation to dictionary for JSON serialization."""
        return {
            'factor_name': factor.factor_name,
            'factor_type': factor.factor_type,
            'confidence': factor.confidence,
            'description': factor.description,
            'top_loadings': factor.top_loadings,
            'correlation_with_benchmark': factor.correlation_with_benchmark,
            'variance_explained': factor.variance_explained
        }


class PortfolioCorrelationAnalyzer:
    """
    Main analyzer for portfolio correlation structure analysis.
    
    Combines effective rank analysis, Marchenko-Pastur validation,
    and risk concentration metrics into comprehensive portfolio insights.
    """
    
    def __init__(self):
        self.mp_analyzer = MarchenkoPosturAnalyzer()
        self.tail_risk_analyzer = TailRiskAnalyzer()
        self.factor_analyzer = FactorAttributionAnalyzer()
        self.data_service = DataService()
    
    def analyze_portfolio_returns(self, returns_data: pd.DataFrame, 
                                holdings_weights: Optional[pd.Series] = None,
                                include_tail_risk: bool = False,
                                include_factor_attribution: bool = False) -> Dict[str, Any]:
        """
        Perform portfolio correlation analysis with optional tail risk and factor attribution.
        
        Args:
            returns_data: DataFrame with asset returns
            holdings_weights: Optional Series with asset weights
            include_tail_risk: Whether to include tail risk analysis
            include_factor_attribution: Whether to include factor attribution
            
        Returns:
            Complete analysis results dictionary
        """
        # Validate data
        is_valid, error_msg = validate_correlation_data(returns_data)
        if not is_valid:
            return {'error': error_msg, 'analysis_successful': False}
        
        try:
            # Calculate correlation matrix
            correlation_matrix = returns_data.corr()
            correlation_matrix = clean_correlation_matrix(correlation_matrix)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = perform_eigenvalue_decomposition(correlation_matrix)
            
            N = len(eigenvalues)
            T = returns_data.shape[0]
            
            # Core analysis components
            analysis_results = {
                'analysis_successful': True,
                'data_info': {
                    'num_assets': N,
                    'num_observations': T,
                    'date_range': {
                        'start': returns_data.index.min().strftime('%Y-%m-%d'),
                        'end': returns_data.index.max().strftime('%Y-%m-%d')
                    }
                }
            }
            
            # 1. Effective Rank Analysis
            effective_rank_results = calculate_effective_rank_metrics(eigenvalues, N, holdings_weights)
            analysis_results.update(effective_rank_results)
            
            # 2. Marchenko-Pastur Analysis
            mp_results = self._perform_mp_analysis_safe(eigenvalues, N, T)
            analysis_results.update(mp_results)
            
            # 3. Risk Concentration Analysis
            risk_analysis = analyze_risk_concentration(eigenvalues, correlation_matrix, holdings_weights)
            analysis_results.update(risk_analysis)
            
            # 4. Generate interpretations
            interpretation = generate_correlation_interpretation(
                effective_rank_results, risk_analysis, mp_results
            )
            analysis_results['interpretation'] = interpretation
            
            # 5. Generate advisor talking points
            talking_points = generate_advisor_talking_points(effective_rank_results, mp_results)
            analysis_results['advisor_talking_points'] = talking_points
            
            # 6. Add quality metrics
            analysis_results['quality_metrics'] = self._calculate_quality_metrics(
                returns_data, correlation_matrix, mp_results
            )
            
            # 7. Optional tail risk analysis
            if include_tail_risk:
                tail_risk_results = self.tail_risk_analyzer.analyze_tail_risk(returns_data, holdings_weights)
                if tail_risk_results.get('analysis_successful', False):
                    analysis_results['tail_risk_analysis'] = tail_risk_results
                    analysis_results['enhanced_interpretation'] = self._generate_enhanced_interpretation(
                        analysis_results, tail_risk_results
                    )
                    analysis_results['enhanced_advisor_points'] = self._generate_enhanced_advisor_points(
                        analysis_results, tail_risk_results
                    )
            
            # 8. Optional factor attribution analysis
            if include_factor_attribution:
                factor_results = self.factor_analyzer.analyze_factor_attribution(
                    returns_data, eigenvalues, eigenvectors
                )
                if factor_results.get('attribution_successful', False):
                    analysis_results['factor_attribution'] = factor_results
                    analysis_results['factor_enhanced_interpretation'] = self._generate_factor_enhanced_interpretation(
                        analysis_results, factor_results
                    )
            
            return analysis_results
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'analysis_successful': False
            }
    
    def _perform_mp_analysis_safe(self, eigenvalues: np.ndarray, N: int, T: int) -> Dict[str, Any]:
        """Safely perform Marchenko-Pastur analysis with error handling."""
        try:
            mp_results = self.mp_analyzer.analyze(eigenvalues, N, T)
            
            # Add validation metrics
            if mp_results.get('mp_fitting_successful', False):
                mp_results['mp_validation'] = self._validate_mp_results(mp_results)
            
            return mp_results
            
        except Exception as e:
            return {
                'mp_fitting_successful': False,
                'mp_error': str(e),
                'fallback_reason': 'MP analysis failed - using effective rank only'
            }
    
    def _validate_mp_results(self, mp_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MP results against expected patterns."""
        q_ratio = mp_results['q_ratio']
        noise_fraction = mp_results['noise_fraction']
        num_signal = mp_results['num_signal_factors']
        
        validation = {
            'q_ratio_acceptable': q_ratio < 0.5,
            'noise_level_reasonable': 0.3 <= noise_fraction <= 0.95,
            'signal_factors_reasonable': 1 <= num_signal <= mp_results['num_eigenvalues'] // 2,
            'largest_eigenvalue_significant': mp_results.get('largest_signal_eigenvalue', 0) > 2.0
        }
        
        validation['overall_credible'] = sum(validation.values()) >= 3
        validation['credibility_score'] = sum(validation.values()) / len(validation)
        
        return validation
    
    def _calculate_quality_metrics(self, returns_data: pd.DataFrame,
                                  correlation_matrix: pd.DataFrame,
                                  mp_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall analysis quality metrics."""
        # Data quality
        data_quality = {
            'sufficient_observations': returns_data.shape[0] >= returns_data.shape[1] * 3,
            'low_missing_data': returns_data.isnull().sum().sum() / returns_data.size < 0.05,
            'reasonable_correlations': (correlation_matrix.abs() < 0.98).all().all()
        }
        
        # Analysis quality
        analysis_quality = {
            'mp_analysis_available': mp_results.get('mp_fitting_successful', False),
            'mp_credible': mp_results.get('mp_validation', {}).get('overall_credible', False) if mp_results.get('mp_fitting_successful') else False
        }
        
        # Overall score
        all_checks = {**data_quality, **analysis_quality}
        overall_score = sum(all_checks.values()) / len(all_checks)
        
        return {
            'data_quality': data_quality,
            'analysis_quality': analysis_quality,
            'overall_score': overall_score,
            'quality_level': 'High' if overall_score >= 0.8 else 'Medium' if overall_score >= 0.6 else 'Low'
        }
    
    def _generate_enhanced_interpretation(self, correlation_results: Dict[str, Any], 
                                        tail_risk_results: Dict[str, Any]) -> List[str]:
        """Generate enhanced interpretation combining correlation and tail risk insights."""
        interpretation = []
        
        # Executive summary combining both analyses
        effective_rank = correlation_results.get('effective_rank', 0)
        num_assets = correlation_results.get('num_assets', 0)
        diversification_loss = correlation_results.get('diversification_loss', 0)
        max_drawdown = abs(tail_risk_results['drawdown_analysis']['maximum_drawdown'])
        var_99 = abs(tail_risk_results['var_cvar_analysis']['99.0%']['historical_var_daily'])
        
        interpretation.append(f"Comprehensive Risk Assessment ({num_assets} assets):")
        interpretation.append(f"• Portfolio concentration: {effective_rank:.1f} effective assets ({diversification_loss:.0%} diversification loss)")
        interpretation.append(f"• Historical tail risk: {max_drawdown:.1%} maximum drawdown, {var_99:.2%} daily 99% VaR")
        
        # Risk correlation insights
        if diversification_loss > 0.6 and max_drawdown > 0.3:
            interpretation.append("• High correlation amplifies tail risk - systematic vulnerability detected")
        elif diversification_loss > 0.4 and max_drawdown > 0.2:
            interpretation.append("• Moderate correlation increases downside beyond individual asset risk")
        elif diversification_loss < 0.3 and max_drawdown < 0.15:
            interpretation.append("• Good diversification provides effective tail risk protection")
        
        return interpretation
    
    def _generate_enhanced_advisor_points(self, correlation_results: Dict[str, Any],
                                        tail_risk_results: Dict[str, Any]) -> List[str]:
        """Generate enhanced advisor talking points combining both analyses."""
        talking_points = []
        
        # Key risk headline combining correlation and tail risk
        effective_rank = correlation_results.get('effective_rank', 0)
        num_assets = correlation_results.get('num_assets', 0)
        max_drawdown = abs(tail_risk_results['drawdown_analysis']['maximum_drawdown'])
        var_95 = abs(tail_risk_results['var_cvar_analysis']['95.0%']['historical_var_daily'])
        
        talking_points.append(f"Portfolio Reality Check: {num_assets} holdings behave like {effective_rank:.1f} independent investments")
        talking_points.append(f"Downside Protection: 95% confident daily losses stay below {var_95:.1%}, worst historical loss {max_drawdown:.1%}")
        
        # Correlation-risk connection
        diversification_loss = correlation_results.get('diversification_loss', 0)
        if diversification_loss > 0.5:
            talking_points.append(f"Hidden Risk: {diversification_loss:.0%} diversification loss means correlation concentrates risk")
        
        talking_points.append("Analysis Method: Eigenvalue decomposition + tail risk metrics provide institutional-grade risk assessment")
        
        return talking_points
    
    def _generate_factor_enhanced_interpretation(self, correlation_results: Dict[str, Any],
                                               factor_attribution_results: Dict[str, Any]) -> List[str]:
        """Generate interpretation enhanced with factor insights."""
        interpretation = []
        
        factor_interpretations = factor_attribution_results.get('factor_interpretations', [])
        if not factor_interpretations:
            return interpretation
        
        # Factor-driven portfolio narrative
        effective_rank = correlation_results.get('effective_rank', 0)
        num_assets = correlation_results.get('num_assets', 0)
        
        interpretation.append(f"Factor-Driven Portfolio Analysis ({num_assets} assets):")
        
        # Top 3 factors with economic meaning
        for i, factor in enumerate(factor_interpretations[:3]):
            factor_desc = f"Factor {i+1}: {factor['factor_name']} ({factor['variance_explained']:.1%} variance)"
            interpretation.append(f"• {factor_desc}")
            interpretation.append(f"  - {factor['description']}")
            
            # Show top holdings for this factor
            if factor['top_loadings']:
                top_stocks = [f"{ticker} ({loading:.2f})" for ticker, loading in factor['top_loadings'][:3]]
                interpretation.append(f"  - Key exposures: {', '.join(top_stocks)}")
        
        return interpretation


# Enhanced Portfolio Analyzer that combines everything
class EnhancedPortfolioAnalyzer:
    """
    Enhanced portfolio analyzer combining correlation structure and tail risk analysis.
    
    Provides comprehensive portfolio risk assessment including:
    - Correlation structure analysis
    - Tail risk metrics (VaR, CVaR, drawdowns)
    - Stress testing
    - Integrated risk interpretation
    """
    
    def __init__(self):
        self.mp_analyzer = MarchenkoPosturAnalyzer()
        self.tail_risk_analyzer = TailRiskAnalyzer()
        self.factor_analyzer = FactorAttributionAnalyzer()
        self.data_service = DataService()
    
    def analyze_comprehensive_portfolio_risk(self, returns_data: pd.DataFrame, 
                                           holdings_weights: Optional[pd.Series] = None,
                                           include_tail_risk: bool = True,
                                           include_factor_attribution: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio risk analysis including correlation, tail risk, and optionally factors.
        
        Args:
            returns_data: DataFrame with asset returns
            holdings_weights: Optional Series with asset weights
            include_tail_risk: Whether to include tail risk analysis
            include_factor_attribution: Whether to include factor attribution
            
        Returns:
            Complete risk analysis results dictionary
        """
        # Use the base analyzer with all features enabled
        analyzer = PortfolioCorrelationAnalyzer()
        return analyzer.analyze_portfolio_returns(
            returns_data, holdings_weights, include_tail_risk, include_factor_attribution
        )
    
    # Preserve all existing methods for backward compatibility
    def _perform_correlation_analysis(self, returns_data: pd.DataFrame, 
                                    holdings_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Perform the existing correlation analysis."""
        analyzer = PortfolioCorrelationAnalyzer()
        return analyzer.analyze_portfolio_returns(returns_data, holdings_weights, 
                                                include_tail_risk=False, include_factor_attribution=False)
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                    report_type: str = "comprehensive") -> str:
        """Generate comprehensive report combining all analyses."""
        if not analysis_results.get('analysis_successful', False):
            return f"Analysis Failed: {analysis_results.get('error', 'Unknown error')}"
        
        if report_type == "executive_summary":
            return self._generate_executive_summary(analysis_results)
        elif report_type == "advisor_presentation":
            return self._generate_advisor_presentation(analysis_results)
        elif report_type == "risk_committee":
            return self._generate_risk_committee_report(analysis_results)
        else:  # comprehensive
            return self._generate_comprehensive_report(analysis_results)
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary combining key insights."""
        report = "EXECUTIVE PORTFOLIO RISK SUMMARY\n"
        report += "=" * 35 + "\n\n"
        
        # Key metrics
        report += f"Portfolio Size: {results['num_assets']} assets\n"
        report += f"Effective Diversification: {results['effective_rank']:.1f} independent positions\n"
        report += f"Diversification Loss: {results['diversification_loss']:.0%}\n"
        
        if 'tail_risk_analysis' in results:
            tail = results['tail_risk_analysis']
            report += f"Maximum Historical Loss: {abs(tail['drawdown_analysis']['maximum_drawdown']):.1%}\n"
            report += f"99% Daily Value at Risk: {abs(tail['var_cvar_analysis']['99.0%']['historical_var_daily']):.2%}\n"
        
        report += "\nKEY INSIGHTS:\n"
        if 'enhanced_interpretation' in results:
            for insight in results['enhanced_interpretation'][:3]:
                report += f"• {insight}\n"
        elif 'interpretation' in results:
            for insight in results['interpretation'][:3]:
                report += f"• {insight}\n"
        
        return report
    
    def _generate_advisor_presentation(self, results: Dict[str, Any]) -> str:
        """Generate advisor presentation format."""
        report = "CLIENT PORTFOLIO RISK PRESENTATION\n"
        report += "=" * 35 + "\n\n"
        
        if 'enhanced_advisor_points' in results:
            for point in results['enhanced_advisor_points']:
                report += f"• {point}\n\n"
        elif 'advisor_talking_points' in results:
            for point in results['advisor_talking_points']:
                report += f"• {point}\n\n"
        
        return report
    
    def _generate_risk_committee_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed risk committee report."""
        report = "RISK COMMITTEE PORTFOLIO ANALYSIS\n"
        report += "=" * 40 + "\n\n"
        
        # Data overview
        report += "DATA SUMMARY:\n"
        data_info = results['data_info']
        report += f"Period: {data_info['date_range']['start']} to {data_info['date_range']['end']}\n"
        report += f"Assets: {data_info['num_assets']}, Observations: {data_info['num_observations']}\n\n"
        
        # Correlation analysis
        report += "CORRELATION STRUCTURE:\n"
        report += f"Effective Rank: {results['effective_rank']:.2f}\n"
        report += f"Diversification Loss: {results['diversification_loss']:.1%}\n"
        if results.get('mp_fitting_successful'):
            report += f"Signal Factors: {results['num_signal_factors']}\n"
            report += f"Noise Fraction: {results['noise_fraction']:.1%}\n"
        report += "\n"
        
        # Tail risk analysis
        if 'tail_risk_analysis' in results:
            tail = results['tail_risk_analysis']
            report += "TAIL RISK METRICS:\n"
            report += f"Maximum Drawdown: {abs(tail['drawdown_analysis']['maximum_drawdown']):.1%}\n"
            report += f"95% VaR (daily): {abs(tail['var_cvar_analysis']['95.0%']['historical_var_daily']):.2%}\n"
            report += f"99% CVaR (daily): {abs(tail['var_cvar_analysis']['99.0%']['historical_cvar_daily']):.2%}\n"
        
        # Quality assessment
        quality = results['quality_metrics']
        report += f"\nAnalysis Quality: {quality['quality_level']} (Score: {quality['overall_score']:.2f})\n"
        
        return report
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed comprehensive report."""
        report = "COMPREHENSIVE PORTFOLIO RISK ANALYSIS\n"
        report += "=" * 45 + "\n\n"
        
        # Executive summary
        report += self._generate_executive_summary(results) + "\n\n"
        
        # Detailed interpretation
        if 'enhanced_interpretation' in results:
            report += "DETAILED ANALYSIS:\n"
            for line in results['enhanced_interpretation']:
                report += f"{line}\n"
            report += "\n"
        elif 'interpretation' in results:
            report += "CORRELATION ANALYSIS:\n"
            for line in results['interpretation']:
                report += f"{line}\n"
            report += "\n"
        
        # Factor analysis
        if 'factor_attribution' in results:
            report += "FACTOR ATTRIBUTION:\n"
            for insight in results['factor_attribution']['economic_insights']:
                report += f"• {insight}\n"
            report += "\n"
        
        # Methodology
        report += "METHODOLOGY:\n"
        report += "• Eigenvalue decomposition of correlation matrix\n"
        report += "• Random Matrix Theory validation (Marchenko-Pastur)\n"
        if 'tail_risk_analysis' in results:
            report += "• Historical simulation Value at Risk\n"
            report += "• Stress testing against major market events\n"
        if 'factor_attribution' in results:
            report += "• Factor attribution with economic interpretation\n"
        
        return report


# Convenience functions for analysis
def analyze_portfolio_returns(returns_df: pd.DataFrame, 
                             weights: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Quick analysis of returns data with correlation analysis only.
    
    Args:
        returns_df: DataFrame with returns data
        weights: Optional weights series
        
    Returns:
        Analysis results dictionary
    """
    analyzer = PortfolioCorrelationAnalyzer()
    return analyzer.analyze_portfolio_returns(returns_df, weights)


def analyze_comprehensive_portfolio_risk(returns_df: pd.DataFrame, 
                                       weights: Optional[pd.Series] = None,
                                       include_tail_risk: bool = True,
                                       include_factor_attribution: bool = False) -> Dict[str, Any]:
    """
    Comprehensive portfolio risk analysis.
    
    Args:
        returns_df: DataFrame with returns data
        weights: Optional weights series
        include_tail_risk: Whether to include tail risk analysis
        include_factor_attribution: Whether to include factor attribution
        
    Returns:
        Comprehensive analysis results dictionary
    """
    analyzer = EnhancedPortfolioAnalyzer()
    return analyzer.analyze_comprehensive_portfolio_risk(
        returns_df, weights, include_tail_risk, include_factor_attribution
    )
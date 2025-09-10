"""
Factor attribution analysis module for portfolio correlation analysis.

This module maps eigenvalue factors to economic interpretations by analyzing
factor loadings and correlating them with known market factors, sectors,
and style characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime, timedelta


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
        # Reference factors for interpretation
        self.reference_factors = {
            'market': {
                'SPY': 'S&P 500 Market Factor',
                'QQQ': 'NASDAQ Technology Factor',
                'IWM': 'Small Cap Factor'
            },
            'sector': {
                'XLF': 'Financial Sector',
                'XLT': 'Technology Sector', 
                'XLE': 'Energy Sector',
                'XLV': 'Healthcare Sector',
                'XLI': 'Industrial Sector',
                'XLK': 'Technology Sector',
                'XLP': 'Consumer Staples',
                'XLY': 'Consumer Discretionary',
                'XLB': 'Materials Sector',
                'XLU': 'Utilities Sector',
                'XLRE': 'Real Estate Sector'
            },
            'style': {
                'IWF': 'Large Cap Growth',
                'IWD': 'Large Cap Value',
                'IWO': 'Small Cap Growth',
                'IWN': 'Small Cap Value',
                'MTUM': 'Momentum Factor',
                'QUAL': 'Quality Factor',
                'USMV': 'Low Volatility Factor'
            }
        }
        
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
            
            # Calculate factor time series
            factor_returns = self._calculate_factor_returns(
                returns_data, eigenvectors, top_factor_indices
            )
            
            # Analyze factor stability over time
            stability_analysis = self._analyze_factor_stability(
                returns_data, factor_returns
            )
            
            # Generate sector exposure analysis
            sector_exposure = self._analyze_sector_exposure(
                returns_data.columns, eigenvectors, top_factor_indices
            )
            
            return {
                'attribution_successful': True,
                'num_factors_analyzed': num_factors,
                'factor_interpretations': [self._factor_to_dict(f) for f in factor_interpretations],
                'factor_returns': factor_returns,
                'stability_analysis': stability_analysis,
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
        """
        Interpret a single factor by analyzing its loadings and correlations.
        
        Args:
            returns_data: Portfolio returns data
            eigenvector: Factor loadings (eigenvector)
            eigenvalue: Factor importance (eigenvalue)
            factor_number: Factor rank by importance
            
        Returns:
            FactorInterpretation object
        """
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
        """
        Classify factor based on loading patterns and ticker analysis.
        
        Returns:
            Tuple of (factor_type, factor_name, confidence, description)
        """
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
        
        elif np.std(loadings) < 0.2:
            # Low variance in loadings - equal weight factor
            return 'style', 'Equal Weight Factor', 0.6, 'Factor with relatively equal influence across assets'
        
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
        """
        Estimate correlation with market factor based on loading uniformity.
        High uniform positive loadings suggest market factor.
        """
        # Market factors typically have positive, relatively uniform loadings
        mean_loading = np.mean(loadings)
        loading_std = np.std(loadings)
        positive_ratio = np.sum(loadings > 0) / len(loadings)
        
        # High positive mean, low std, high positive ratio = market-like
        market_score = (mean_loading * positive_ratio) / (1 + loading_std)
        
        # Normalize to [-1, 1]
        return float(np.tanh(market_score * 2))
    
    def _calculate_factor_returns(self, returns_data: pd.DataFrame,
                                 eigenvectors: np.ndarray,
                                 factor_indices: np.ndarray) -> pd.DataFrame:
        """
        Calculate factor return time series.
        
        Args:
            returns_data: Asset returns
            eigenvectors: All eigenvectors
            factor_indices: Indices of factors to calculate
            
        Returns:
            DataFrame with factor returns over time
        """
        factor_returns = pd.DataFrame(index=returns_data.index)
        
        for i, factor_idx in enumerate(factor_indices):
            factor_loadings = eigenvectors[:, factor_idx]
            
            # Calculate factor returns as weighted sum of asset returns
            factor_return = (returns_data * factor_loadings).sum(axis=1)
            factor_returns[f'Factor_{i+1}'] = factor_return
        
        return factor_returns
    
    def _analyze_factor_stability(self, returns_data: pd.DataFrame,
                                 factor_returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze stability of factors over time using rolling windows.
        """
        stability_metrics = {}
        
        for factor_col in factor_returns.columns:
            factor_series = factor_returns[factor_col]
            
            # Rolling volatility (21-day window)
            rolling_vol = factor_series.rolling(21).std()
            
            # Regime changes (periods of high vs low volatility)
            vol_percentiles = rolling_vol.quantile([0.25, 0.75])
            high_vol_periods = (rolling_vol > vol_percentiles[0.75]).sum()
            low_vol_periods = (rolling_vol < vol_percentiles[0.25]).sum()
            
            stability_metrics[factor_col] = {
                'average_volatility': float(factor_series.std()),
                'volatility_stability': float(1 / (1 + rolling_vol.std())),  # Higher = more stable
                'high_volatility_periods': int(high_vol_periods),
                'low_volatility_periods': int(low_vol_periods),
                'current_volatility_percentile': float(rolling_vol.iloc[-21:].mean() / rolling_vol.mean()) if len(rolling_vol) > 21 else 1.0
            }
        
        return stability_metrics
    
    def _analyze_sector_exposure(self, tickers: List[str],
                                eigenvectors: np.ndarray,
                                factor_indices: np.ndarray) -> Dict[str, Any]:
        """
        Analyze sector exposure for each factor.
        """
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
        if 'style' in type_counts:
            summary.append(f"• {type_counts['style']} style factor(s) - investment characteristics")
        
        # Factor concentration
        total_variance = sum(f.variance_explained for f in interpretations)
        summary.append(f"Top factors explain {total_variance:.1%} of portfolio variance")
        
        # Dominant factor
        if interpretations:
            dominant = interpretations[0]
            summary.append(f"Dominant factor: {dominant.factor_name} ({dominant.variance_explained:.1%} of variance)")
        
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
        
        # Idiosyncratic risk insights
        idio_factors = [f for f in interpretations if f.factor_type == 'idiosyncratic']
        if len(idio_factors) > 1:
            insights.append("Multiple stock-specific factors - individual positions drive portfolio risk")
        
        # Diversification insights
        if len(interpretations) <= 2:
            insights.append("Few dominant factors - portfolio lacks diversification across risk sources")
        elif len(interpretations) >= 4:
            insights.append("Multiple risk factors - good diversification across different sources of risk")
        
        # Factor stability insights
        high_confidence_factors = [f for f in interpretations if f.confidence > 0.7]
        if len(high_confidence_factors) > len(interpretations) / 2:
            insights.append("Clear factor structure - portfolio exposures are well-defined and interpretable")
        else:
            insights.append("Mixed factor structure - portfolio has complex, overlapping risk exposures")
        
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


def integrate_factor_attribution(correlation_results: Dict[str, Any],
                                returns_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Integrate factor attribution with existing correlation analysis.
    
    Args:
        correlation_results: Results from portfolio correlation analysis
        returns_data: Original returns data
        
    Returns:
        Enhanced results with factor attribution
    """
    try:
        # Extract eigenvalue decomposition results
        eigenvalues = np.array(correlation_results.get('eigenvalues', []))
        
        # Reconstruct eigenvectors from correlation matrix (simplified approach)
        # In practice, you'd want to store eigenvectors from the original analysis
        correlation_matrix = returns_data.corr()
        eigenvalues_calc, eigenvectors = np.linalg.eigh(correlation_matrix.values)
        
        # Sort in descending order (matching original analysis)
        idx = eigenvalues_calc.argsort()[::-1]
        eigenvalues_calc = eigenvalues_calc[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Perform factor attribution
        factor_analyzer = FactorAttributionAnalyzer()
        attribution_results = factor_analyzer.analyze_factor_attribution(
            returns_data, eigenvalues_calc, eigenvectors
        )
        
        # Combine with original correlation results
        enhanced_results = correlation_results.copy()
        enhanced_results['factor_attribution'] = attribution_results
        
        # Add integrated insights
        if attribution_results.get('attribution_successful', False):
            enhanced_results['factor_insights'] = _generate_integrated_factor_insights(
                correlation_results, attribution_results
            )
        
        return enhanced_results
        
    except Exception as e:
        # Return original results with error note
        enhanced_results = correlation_results.copy()
        enhanced_results['factor_attribution'] = {
            'attribution_successful': False,
            'error': str(e)
        }
        return enhanced_results


def _generate_integrated_factor_insights(correlation_results: Dict[str, Any],
                                        attribution_results: Dict[str, Any]) -> List[str]:
    """Generate insights combining correlation and factor attribution analysis."""
    insights = []
    
    effective_rank = correlation_results.get('effective_rank', 0)
    num_factors = attribution_results.get('num_factors_analyzed', 0)
    
    # Correlation vs factor count relationship
    if effective_rank < 3 and num_factors >= 2:
        insights.append(f"Low effective rank ({effective_rank:.1f}) despite {num_factors} factors - strong inter-factor correlation")
    elif effective_rank > num_factors * 0.8:
        insights.append(f"Effective rank ({effective_rank:.1f}) close to factor count ({num_factors}) - factors are relatively independent")
    
    # Factor concentration insights
    interpretations = attribution_results.get('factor_interpretations', [])
    if interpretations:
        dominant_factor = interpretations[0]
        diversification_loss = correlation_results.get('diversification_loss', 0)
        
        if dominant_factor['variance_explained'] > 0.4 and diversification_loss > 0.5:
            insights.append(f"Dominant {dominant_factor['factor_name']} drives both correlation concentration and diversification loss")
    
    # Market factor vs MP analysis
    if correlation_results.get('mp_fitting_successful', False):
        noise_fraction = correlation_results.get('noise_fraction', 0)
        market_factors = [f for f in interpretations if f['factor_type'] == 'market']
        
        if market_factors and noise_fraction < 0.5:
            insights.append("Strong market factor presence consistent with low noise fraction in MP analysis")
        elif not market_factors and noise_fraction > 0.7:
            insights.append("No clear market factor detected - high noise level suggests weak systematic relationships")
    
    return insights
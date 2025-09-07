"""
Core correlation analysis functions for portfolio diversification analysis.

This module provides eigenvalue-based analysis of portfolio correlation structures,
including effective rank calculations and risk concentration metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def calculate_effective_rank_metrics(eigenvalues: np.ndarray, N: int, 
                                   holdings_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Calculate effective rank and diversification metrics from eigenvalues.
    
    Args:
        eigenvalues: Array of correlation matrix eigenvalues
        N: Number of assets
        holdings_weights: Optional asset weights (not currently used)
        
    Returns:
        Dictionary containing effective rank metrics
    """
    # Normalize eigenvalues to sum to N (trace of correlation matrix)
    eigenvalue_weights = eigenvalues / np.sum(eigenvalues)
    
    # Effective rank (participation ratio)
    effective_rank = 1 / np.sum(eigenvalue_weights**2)
    
    # Diversification metrics
    concentration_ratio = effective_rank / N
    diversification_loss = 1 - concentration_ratio
    
    # Factor contributions
    factor_contributions = eigenvalue_weights[:min(5, len(eigenvalue_weights))]
    
    return {
        'num_assets': N,
        'effective_rank': float(effective_rank),
        'concentration_ratio': float(concentration_ratio),
        'diversification_loss': float(diversification_loss),
        'eigenvalues': eigenvalues.tolist(),
        'eigenvalue_weights': eigenvalue_weights.tolist(),
        'factor_contributions': factor_contributions.tolist(),
        'largest_eigenvalue_weight': float(eigenvalue_weights[0]),
        'top_3_factors_weight': float(np.sum(factor_contributions[:3])) if len(factor_contributions) >= 3 else float(np.sum(factor_contributions))
    }


def analyze_risk_concentration(eigenvalues: np.ndarray, correlation_matrix: pd.DataFrame,
                              holdings_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Analyze risk concentration patterns in the portfolio.
    
    Args:
        eigenvalues: Array of correlation matrix eigenvalues
        correlation_matrix: Full correlation matrix
        holdings_weights: Optional asset weights
        
    Returns:
        Dictionary containing risk concentration metrics
    """
    eigenvalue_weights = eigenvalues / np.sum(eigenvalues)
    
    # Concentration measures
    herfindahl_index = np.sum(eigenvalue_weights**2)
    entropy = -np.sum(eigenvalue_weights * np.log(eigenvalue_weights + 1e-10))
    max_entropy = np.log(len(eigenvalues))
    normalized_entropy = entropy / max_entropy
    
    # Correlation structure analysis
    off_diagonal = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
    avg_correlation = np.mean(off_diagonal)
    correlation_std = np.std(off_diagonal)
    high_correlations = np.sum(off_diagonal > 0.7)
    
    return {
        'risk_concentration': {
            'herfindahl_index': float(herfindahl_index),
            'entropy_normalized': float(normalized_entropy),
            'concentration_level': 'High' if herfindahl_index > 0.3 else 'Moderate' if herfindahl_index > 0.15 else 'Low'
        },
        'correlation_structure': {
            'average_correlation': float(avg_correlation),
            'correlation_std': float(correlation_std),
            'num_high_correlations': int(high_correlations),
            'correlation_pattern': 'Highly correlated' if avg_correlation > 0.6 else 'Moderately correlated' if avg_correlation > 0.3 else 'Weakly correlated'
        }
    }


def clean_correlation_matrix(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Clean correlation matrix for numerical stability.
    
    Args:
        corr_matrix: Raw correlation matrix
        
    Returns:
        Cleaned correlation matrix
    """
    # Replace any NaN with 0 correlation
    corr_matrix = corr_matrix.fillna(0)
    
    # Ensure diagonal is 1
    np.fill_diagonal(corr_matrix.values, 1.0)
    
    # Ensure symmetry
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    
    # Clamp off-diagonal values to valid correlation range
    corr_matrix = corr_matrix.clip(-0.99, 0.99)
    np.fill_diagonal(corr_matrix.values, 1.0)
    
    return corr_matrix


def validate_correlation_data(returns_data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate returns data for correlation analysis.
    
    Args:
        returns_data: DataFrame of asset returns
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if returns_data.empty:
        return False, 'Empty returns data provided'
    
    # Check for sufficient observations
    min_observations = max(50, returns_data.shape[1] * 2)
    if returns_data.shape[0] < min_observations:
        return False, f'Insufficient data: {returns_data.shape[0]} obs, need {min_observations}'
    
    # Check for NaN values
    if returns_data.isnull().any().any():
        return False, 'Returns data contains NaN values'
    
    # Check for constant columns (zero variance)
    constant_cols = returns_data.std() == 0
    if constant_cols.any():
        return False, f'Assets with zero variance: {constant_cols.sum()}'
    
    return True, 'Valid'


def perform_eigenvalue_decomposition(correlation_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform eigenvalue decomposition on correlation matrix.
    
    Args:
        correlation_matrix: Cleaned correlation matrix
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted in descending order
    """
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix.values)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Remove numerical zeros
    positive_mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[positive_mask]
    eigenvectors = eigenvectors[:, positive_mask]
    
    return eigenvalues, eigenvectors


def generate_correlation_interpretation(effective_rank_metrics: Dict[str, Any], 
                                       risk_concentration: Dict[str, Any],
                                       mp_results: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Generate comprehensive interpretation of correlation analysis results.
    
    Args:
        effective_rank_metrics: Results from effective rank analysis
        risk_concentration: Results from risk concentration analysis
        mp_results: Optional Marchenko-Pastur results
        
    Returns:
        List of interpretation strings
    """
    interpretation = []
    
    # Effective rank insights
    effective_rank = effective_rank_metrics['effective_rank']
    diversification_loss = effective_rank_metrics['diversification_loss']
    num_assets = effective_rank_metrics['num_assets']
    
    interpretation.append(f"Portfolio Diversification Analysis ({num_assets} assets):")
    interpretation.append(f"• Effective rank: {effective_rank:.1f} (behaves like {effective_rank:.1f} independent assets)")
    interpretation.append(f"• Diversification loss: {diversification_loss:.1%} (concentrated vs. equal-weight ideal)")
    
    # Factor concentration
    largest_factor = effective_rank_metrics['largest_eigenvalue_weight']
    top_3_factors = effective_rank_metrics['top_3_factors_weight']
    
    interpretation.append(f"• Largest factor explains {largest_factor:.1%} of portfolio variance")
    interpretation.append(f"• Top 3 factors explain {top_3_factors:.1%} of portfolio variance")
    
    # Marchenko-Pastur insights
    if mp_results and mp_results.get('mp_fitting_successful', False):
        noise_fraction = mp_results['noise_fraction']
        num_signal = mp_results['num_signal_factors']
        
        interpretation.append(f"Random Matrix Theory Validation:")
        interpretation.append(f"• {noise_fraction:.1%} of correlations are random noise")
        interpretation.append(f"• {num_signal} genuine risk factors identified")
    
    # Risk concentration insights
    if risk_concentration:
        concentration_level = risk_concentration['risk_concentration']['concentration_level']
        correlation_pattern = risk_concentration['correlation_structure']['correlation_pattern']
        
        interpretation.append(f"Risk Structure:")
        interpretation.append(f"• Concentration level: {concentration_level}")
        interpretation.append(f"• Asset correlations: {correlation_pattern}")
    
    return interpretation


def generate_advisor_talking_points(effective_rank_metrics: Dict[str, Any],
                                   mp_results: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Generate advisor-ready talking points.
    
    Args:
        effective_rank_metrics: Results from effective rank analysis
        mp_results: Optional Marchenko-Pastur results
        
    Returns:
        List of talking point strings
    """
    talking_points = []
    
    effective_rank = effective_rank_metrics['effective_rank']
    diversification_loss = effective_rank_metrics['diversification_loss']
    num_assets = effective_rank_metrics['num_assets']
    largest_factor = effective_rank_metrics['largest_eigenvalue_weight']
    
    # Key headline
    talking_points.append(f"Key Finding: Portfolio acts like {effective_rank:.1f} independent investments despite holding {num_assets} assets")
    
    # Diversification story
    talking_points.append(f"Hidden Cost: {diversification_loss:.0%} diversification loss due to correlation concentration")
    
    # Risk factor story
    talking_points.append(f"Risk Concentration: Single largest factor drives {largest_factor:.0%} of portfolio volatility")
    
    # MP validation if available
    if mp_results and mp_results.get('mp_fitting_successful', False):
        noise_fraction = mp_results['noise_fraction']
        talking_points.append(f"Scientific Validation: Random Matrix Theory confirms {noise_fraction:.0%} of correlations are meaningless noise")
    
    # Actionable insights
    if diversification_loss > 0.6:
        talking_points.append("Recommendation: Portfolio shows extreme concentration - consider broader diversification")
    elif diversification_loss > 0.4:
        talking_points.append("Recommendation: Moderate concentration detected - review factor exposures")
    else:
        talking_points.append("Assessment: Portfolio shows good diversification characteristics")
    
    # Technical credibility
    talking_points.append("Analysis Method: Eigenvalue decomposition with Random Matrix Theory validation")
    
    return talking_points
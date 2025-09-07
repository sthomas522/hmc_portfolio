"""
Main portfolio analyzer that orchestrates correlation analysis.

This module combines all analysis components to provide comprehensive
portfolio diversification analysis with Random Matrix Theory validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

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


class PortfolioCorrelationAnalyzer:
    """
    Main analyzer for portfolio correlation structure analysis.
    
    Combines effective rank analysis, Marchenko-Pastur validation,
    and risk concentration metrics into comprehensive portfolio insights.
    """
    
    def __init__(self):
        self.mp_analyzer = MarchenkoPosturAnalyzer()
        self.data_service = DataService()
    
    def analyze_portfolio_from_csv(self, csv_path: str,
                                  ticker_column: str = 'Ticker',
                                  weight_column: str = 'Portfolio Weight',
                                  period: str = "1y",
                                  max_assets: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze portfolio correlation structure from CSV holdings file.
        
        Args:
            csv_path: Path to CSV file with holdings
            ticker_column: Name of ticker column
            weight_column: Name of weight column
            period: Time period for returns data
            max_assets: Optional limit on number of assets
            
        Returns:
            Complete analysis results dictionary
        """
        # Load holdings
        holdings_df = self.data_service.load_holdings_from_csv(
            csv_path, ticker_column, weight_column
        )
        
        # Fetch returns data
        returns_df, weights_series = self.data_service.fetch_portfolio_returns(
            holdings_df, ticker_column, weight_column, period, max_assets
        )
        
        # Perform analysis
        return self.analyze_portfolio_returns(returns_df, weights_series)
    
    def analyze_portfolio_returns(self, returns_data: pd.DataFrame, 
                                 holdings_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform complete portfolio correlation analysis on returns data.
        
        Args:
            returns_data: DataFrame with asset returns
            holdings_weights: Optional Series with asset weights
            
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
            
            return analysis_results
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'analysis_successful': False
            }
    
    def _perform_mp_analysis_safe(self, eigenvalues: np.ndarray, N: int, T: int) -> Dict[str, Any]:
        """
        Safely perform Marchenko-Pastur analysis with error handling.
        
        Args:
            eigenvalues: Array of eigenvalues
            N: Number of assets
            T: Number of observations
            
        Returns:
            MP analysis results or error information
        """
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
        """
        Validate MP results against expected patterns.
        
        Args:
            mp_results: Results from MP analysis
            
        Returns:
            Validation metrics
        """
        q_ratio = mp_results['q_ratio']
        noise_fraction = mp_results['noise_fraction']
        num_signal = mp_results['num_signal_factors']
        
        validation = {
            'q_ratio_acceptable': q_ratio < 0.5,
            'noise_level_reasonable': 0.3 <= noise_fraction <= 0.95,
            'signal_factors_reasonable': 1 <= num_signal <= mp_results['num_eigenvalues'] // 2,
            'largest_eigenvalue_significant': mp_results.get('largest_signal_eigenvalue', 0) > 2.0
        }
        
        validation['overall_credible'] = sum(validation.values()) >= 3  # At least 3 of 4 checks pass
        validation['credibility_score'] = sum(validation.values()) / len(validation)
        
        return validation
    
    def _calculate_quality_metrics(self, returns_data: pd.DataFrame,
                                  correlation_matrix: pd.DataFrame,
                                  mp_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall analysis quality metrics.
        
        Args:
            returns_data: Original returns data
            correlation_matrix: Correlation matrix
            mp_results: MP analysis results
            
        Returns:
            Quality metrics dictionary
        """
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
    
    def generate_report(self, analysis_results: Dict[str, Any], 
                       report_type: str = "comprehensive") -> str:
        """
        Generate formatted report from analysis results.
        
        Args:
            analysis_results: Results from portfolio analysis
            report_type: Type of report ("comprehensive", "summary", "advisor")
            
        Returns:
            Formatted report string
        """
        if not analysis_results.get('analysis_successful', False):
            return f"Analysis Failed: {analysis_results.get('error', 'Unknown error')}"
        
        if report_type == "summary":
            return self._generate_summary_report(analysis_results)
        elif report_type == "advisor":
            return self._generate_advisor_report(analysis_results)
        else:  # comprehensive
            return self._generate_comprehensive_report(analysis_results)
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate concise summary report."""
        effective_rank = results['effective_rank']
        diversification_loss = results['diversification_loss']
        num_assets = results['num_assets']
        
        report = f"""
PORTFOLIO DIVERSIFICATION SUMMARY
=================================
Assets: {num_assets}
Effective Rank: {effective_rank:.1f}
Diversification Loss: {diversification_loss:.1%}
"""
        
        if results.get('mp_fitting_successful'):
            noise_fraction = results['noise_fraction']
            num_signal = results['num_signal_factors']
            report += f"Random Noise: {noise_fraction:.1%}\n"
            report += f"Signal Factors: {num_signal}\n"
        
        return report.strip()
    
    def _generate_advisor_report(self, results: Dict[str, Any]) -> str:
        """Generate advisor-focused report."""
        report = "ADVISOR PORTFOLIO ANALYSIS\n"
        report += "=" * 30 + "\n\n"
        
        for point in results['advisor_talking_points']:
            report += f"• {point}\n"
        
        return report
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed comprehensive report."""
        report = "COMPREHENSIVE PORTFOLIO CORRELATION ANALYSIS\n"
        report += "=" * 50 + "\n\n"
        
        # Data summary
        data_info = results['data_info']
        report += f"Data Period: {data_info['date_range']['start']} to {data_info['date_range']['end']}\n"
        report += f"Assets: {data_info['num_assets']}, Observations: {data_info['num_observations']}\n\n"
        
        # Main interpretation
        for line in results['interpretation']:
            report += f"{line}\n"
        
        # Quality assessment
        quality = results['quality_metrics']
        report += f"\nAnalysis Quality: {quality['quality_level']} (Score: {quality['overall_score']:.2f})\n"
        
        return report


# Convenience functions for quick analysis
def analyze_csv_portfolio(csv_path: str, **kwargs) -> Dict[str, Any]:
    """
    Quick analysis of portfolio from CSV file.
    
    Args:
        csv_path: Path to CSV holdings file
        **kwargs: Additional arguments for analysis
        
    Returns:
        Analysis results dictionary
    """
    analyzer = PortfolioCorrelationAnalyzer()
    return analyzer.analyze_portfolio_from_csv(csv_path, **kwargs)


def analyze_returns_data(returns_df: pd.DataFrame, 
                        weights: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Quick analysis of returns data.
    
    Args:
        returns_df: DataFrame with returns data
        weights: Optional weights series
        
    Returns:
        Analysis results dictionary
    """
    analyzer = PortfolioCorrelationAnalyzer()
    return analyzer.analyze_portfolio_returns(returns_df, weights)
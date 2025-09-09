"""
Enhanced portfolio analyzer that integrates correlation analysis with tail risk metrics.

This module extends the existing PortfolioCorrelationAnalyzer to include comprehensive
tail risk analysis, providing a complete picture of portfolio risk characteristics.
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
from analysis.tail_risk_analysis import TailRiskAnalyzer, integrate_tail_risk_with_correlation
from data.data_service import DataService


class EnhancedPortfolioAnalyzer:
    """
    Enhanced portfolio analyzer combining correlation structure and tail risk analysis.
    
    Provides comprehensive portfolio risk assessment including:
    - Correlation structure analysis (existing)
    - Tail risk metrics (VaR, CVaR, drawdowns)
    - Stress testing
    - Integrated risk interpretation
    """
    
    def __init__(self):
        self.mp_analyzer = MarchenkoPosturAnalyzer()
        self.tail_risk_analyzer = TailRiskAnalyzer()
        self.data_service = DataService()
    
    def analyze_comprehensive_portfolio_risk(self, returns_data: pd.DataFrame, 
                                           holdings_weights: Optional[pd.Series] = None,
                                           include_tail_risk: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio risk analysis including both correlation and tail risk.
        
        Args:
            returns_data: DataFrame with asset returns
            holdings_weights: Optional Series with asset weights
            include_tail_risk: Whether to include tail risk analysis
            
        Returns:
            Complete risk analysis results dictionary
        """
        # Validate data
        is_valid, error_msg = validate_correlation_data(returns_data)
        if not is_valid:
            return {'error': error_msg, 'analysis_successful': False}
        
        try:
            # 1. Core correlation analysis (existing functionality)
            correlation_results = self._perform_correlation_analysis(returns_data, holdings_weights)
            
            if not correlation_results.get('analysis_successful', False):
                return correlation_results
            
            # 2. Tail risk analysis (new functionality)
            tail_risk_results = {}
            if include_tail_risk:
                tail_risk_results = self.tail_risk_analyzer.analyze_tail_risk(
                    returns_data, holdings_weights
                )
            
            # 3. Integrate analyses
            if include_tail_risk and tail_risk_results.get('analysis_successful', False):
                integrated_results = integrate_tail_risk_with_correlation(
                    correlation_results, tail_risk_results
                )
                
                # Add enhanced interpretations
                integrated_results['enhanced_interpretation'] = self._generate_enhanced_interpretation(
                    correlation_results, tail_risk_results
                )
                integrated_results['enhanced_advisor_points'] = self._generate_enhanced_advisor_points(
                    correlation_results, tail_risk_results
                )
                
                return integrated_results
            else:
                # Return just correlation analysis if tail risk fails or not requested
                return {
                    'correlation_analysis': correlation_results,
                    'tail_risk_analysis': tail_risk_results if include_tail_risk else None,
                    'analysis_type': 'correlation_only'
                }
                
        except Exception as e:
            return {
                'error': f'Enhanced analysis failed: {str(e)}',
                'analysis_successful': False
            }
    
    def _perform_correlation_analysis(self, returns_data: pd.DataFrame, 
                                    holdings_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform the existing correlation analysis.
        
        Args:
            returns_data: DataFrame with asset returns
            holdings_weights: Optional Series with asset weights
            
        Returns:
            Correlation analysis results
        """
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
        """
        Generate enhanced interpretation combining correlation and tail risk insights.
        
        Args:
            correlation_results: Results from correlation analysis
            tail_risk_results: Results from tail risk analysis
            
        Returns:
            List of enhanced interpretation strings
        """
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
        
        # Factor concentration vs stress resilience
        if correlation_results.get('mp_fitting_successful'):
            num_signal_factors = correlation_results.get('num_signal_factors', 0)
            stress_losses = [abs(s['estimated_portfolio_return']) for s in tail_risk_results['stress_testing'].values()]
            avg_stress_loss = np.mean(stress_losses)
            
            interpretation.append(f"• Risk factor analysis: {num_signal_factors} genuine factors, {avg_stress_loss:.1%} average stress scenario loss")
            
            if num_signal_factors <= 2 and avg_stress_loss > 0.25:
                interpretation.append("• Warning: Few risk factors create systematic stress vulnerability")
        
        # Recovery and regime analysis
        recovery_days = tail_risk_results['drawdown_analysis']['recovery_duration_days']
        vol_percentile = tail_risk_results['rolling_volatility']['21_day'].get('volatility_percentile_current')
        
        if recovery_days:
            interpretation.append(f"• Recovery profile: {recovery_days} days to recover from maximum drawdown")
        else:
            interpretation.append("• Recovery profile: Portfolio in prolonged drawdown or slow recovery phase")
        
        if vol_percentile:
            if vol_percentile > 80:
                interpretation.append(f"• Current environment: High volatility regime ({vol_percentile:.0f}th percentile)")
            elif vol_percentile < 20:
                interpretation.append(f"• Current environment: Low volatility regime ({vol_percentile:.0f}th percentile)")
        
        # Tail dependency warning
        if tail_risk_results['tail_dependency']['tail_dependency_available']:
            lower_tail_ratio = tail_risk_results['tail_dependency']['lower_tail_dependency_ratio']
            if lower_tail_ratio > 0.05:
                interpretation.append(f"• Tail correlation: {lower_tail_ratio:.1%} of periods show simultaneous extreme losses")
        
        return interpretation
    
    def _generate_enhanced_advisor_points(self, correlation_results: Dict[str, Any],
                                        tail_risk_results: Dict[str, Any]) -> List[str]:
        """
        Generate enhanced advisor talking points combining both analyses.
        
        Args:
            correlation_results: Results from correlation analysis
            tail_risk_results: Results from tail risk analysis
            
        Returns:
            List of enhanced advisor talking points
        """
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
        
        # Stress test performance with correlation context
        stress_losses = [abs(s['estimated_portfolio_return']) for s in tail_risk_results['stress_testing'].values()]
        worst_stress = max(stress_losses)
        avg_stress = np.mean(stress_losses)
        
        if correlation_results.get('mp_fitting_successful'):
            num_signal_factors = correlation_results.get('num_signal_factors', 0)
            if num_signal_factors <= 2:
                talking_points.append(f"Crisis Vulnerability: Few risk factors ({num_signal_factors}) amplify stress losses to {worst_stress:.1%}")
            else:
                talking_points.append(f"Crisis Resilience: Multiple risk factors ({num_signal_factors}) provide {avg_stress:.1%} average stress loss")
        else:
            talking_points.append(f"Stress Testing: Portfolio estimated to lose {avg_stress:.1%} in major market crises")
        
        # Recovery characteristics with correlation insights
        recovery_days = tail_risk_results['drawdown_analysis']['recovery_duration_days']
        calmar_ratio = tail_risk_results['drawdown_analysis']['calmar_ratio']
        
        if recovery_days and recovery_days > 365:
            talking_points.append(f"Recovery Warning: Historical recovery took {recovery_days} days - correlation slows recovery")
        elif recovery_days:
            talking_points.append(f"Recovery Profile: {recovery_days} days to recover from major losses")
        
        if calmar_ratio > 1:
            talking_points.append(f"Risk-Adjusted Returns: Strong Calmar ratio ({calmar_ratio:.1f}) - good return per unit of drawdown risk")
        elif calmar_ratio < 0.5:
            talking_points.append(f"Risk-Adjusted Returns: Low Calmar ratio ({calmar_ratio:.1f}) - high drawdown relative to returns")
        
        # Current risk environment assessment
        vol_percentile = tail_risk_results['rolling_volatility']['21_day'].get('volatility_percentile_current')
        largest_factor_weight = correlation_results.get('largest_eigenvalue_weight', 0)
        
        if vol_percentile and vol_percentile > 75 and largest_factor_weight > 0.6:
            talking_points.append("Current Risk Alert: High volatility environment amplifies portfolio's factor concentration")
        elif vol_percentile and vol_percentile < 25:
            talking_points.append("Market Opportunity: Low volatility environment reduces impact of correlation concentration")
        
        # Scientific validation and methodology
        if correlation_results.get('mp_fitting_successful'):
            noise_fraction = correlation_results.get('noise_fraction', 0)
            talking_points.append(f"Scientific Validation: Random Matrix Theory confirms {noise_fraction:.0%} correlations are noise, analysis targets genuine risk factors")
        
        talking_points.append("Analysis Method: Eigenvalue decomposition + tail risk metrics provide institutional-grade risk assessment")
        
        return talking_points
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                    report_type: str = "comprehensive") -> str:
        """
        Generate comprehensive report combining correlation and tail risk analysis.
        
        Args:
            analysis_results: Results from comprehensive analysis
            report_type: Type of report to generate
            
        Returns:
            Formatted report string
        """
        if not analysis_results.get('correlation_analysis', {}).get('analysis_successful', False):
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
        corr = results['correlation_analysis']
        tail = results.get('tail_risk_analysis', {})
        
        report = "EXECUTIVE PORTFOLIO RISK SUMMARY\n"
        report += "=" * 35 + "\n\n"
        
        # Key metrics
        report += f"Portfolio Size: {corr['num_assets']} assets\n"
        report += f"Effective Diversification: {corr['effective_rank']:.1f} independent positions\n"
        report += f"Diversification Loss: {corr['diversification_loss']:.0%}\n"
        
        if tail:
            report += f"Maximum Historical Loss: {abs(tail['drawdown_analysis']['maximum_drawdown']):.1%}\n"
            report += f"99% Daily Value at Risk: {abs(tail['var_cvar_analysis']['99.0%']['historical_var_daily']):.2%}\n"
        
        report += "\nKEY INSIGHTS:\n"
        if 'enhanced_interpretation' in results:
            for insight in results['enhanced_interpretation'][:3]:
                report += f"• {insight}\n"
        
        return report
    
    def _generate_advisor_presentation(self, results: Dict[str, Any]) -> str:
        """Generate advisor presentation format."""
        report = "CLIENT PORTFOLIO RISK PRESENTATION\n"
        report += "=" * 35 + "\n\n"
        
        if 'enhanced_advisor_points' in results:
            for point in results['enhanced_advisor_points']:
                report += f"• {point}\n\n"
        
        return report
    
    def _generate_risk_committee_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed risk committee report."""
        corr = results['correlation_analysis']
        tail = results.get('tail_risk_analysis', {})
        
        report = "RISK COMMITTEE PORTFOLIO ANALYSIS\n"
        report += "=" * 40 + "\n\n"
        
        # Data overview
        report += "DATA SUMMARY:\n"
        data_info = corr['data_info']
        report += f"Period: {data_info['date_range']['start']} to {data_info['date_range']['end']}\n"
        report += f"Assets: {data_info['num_assets']}, Observations: {data_info['num_observations']}\n\n"
        
        # Correlation analysis
        report += "CORRELATION STRUCTURE:\n"
        report += f"Effective Rank: {corr['effective_rank']:.2f}\n"
        report += f"Diversification Loss: {corr['diversification_loss']:.1%}\n"
        if corr.get('mp_fitting_successful'):
            report += f"Signal Factors: {corr['num_signal_factors']}\n"
            report += f"Noise Fraction: {corr['noise_fraction']:.1%}\n"
        report += "\n"
        
        # Tail risk analysis
        if tail:
            report += "TAIL RISK METRICS:\n"
            report += f"Maximum Drawdown: {abs(tail['drawdown_analysis']['maximum_drawdown']):.1%}\n"
            report += f"95% VaR (daily): {abs(tail['var_cvar_analysis']['95.0%']['historical_var_daily']):.2%}\n"
            report += f"99% CVaR (daily): {abs(tail['var_cvar_analysis']['99.0%']['historical_cvar_daily']):.2%}\n"
            
            # Stress testing
            report += "\nSTRESS TEST RESULTS:\n"
            for scenario_id, scenario in tail['stress_testing'].items():
                report += f"{scenario['scenario_name']}: {abs(scenario['estimated_portfolio_return']):.1%} loss\n"
        
        # Quality assessment
        quality = corr['quality_metrics']
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
        
        # Integrated insights
        if 'integrated_insights' in results:
            report += "INTEGRATED INSIGHTS:\n"
            for insight in results['integrated_insights']:
                report += f"• {insight}\n"
            report += "\n"
        
        # Methodology
        report += "METHODOLOGY:\n"
        report += "• Eigenvalue decomposition of correlation matrix\n"
        report += "• Random Matrix Theory validation (Marchenko-Pastur)\n"
        report += "• Historical simulation Value at Risk\n"
        report += "• Stress testing against major market events\n"
        report += "• Maximum drawdown and recovery analysis\n"
        
        return report


# Convenience function for comprehensive analysis
def analyze_comprehensive_portfolio_risk(returns_df: pd.DataFrame, 
                                       weights: Optional[pd.Series] = None,
                                       include_tail_risk: bool = True) -> Dict[str, Any]:
    """
    Convenience function for comprehensive portfolio risk analysis.
    
    Args:
        returns_df: DataFrame with returns data
        weights: Optional weights series
        include_tail_risk: Whether to include tail risk analysis
        
    Returns:
        Comprehensive analysis results dictionary
    """
    analyzer = EnhancedPortfolioAnalyzer()
    return analyzer.analyze_comprehensive_portfolio_risk(returns_df, weights, include_tail_risk)
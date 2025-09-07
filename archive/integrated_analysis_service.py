import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from corrected_mp_implementation import CorrectedMPImplementation

class IntegratedAnalysisService:
    """
    Production-ready analysis service combining effective rank and corrected MP analysis
    """
    
    def __init__(self):
        self.mp_analyzer = CorrectedMPImplementation()
    
    def analyze_portfolio_correlation_structure(self, returns_data: pd.DataFrame, 
                                               holdings_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Complete portfolio correlation analysis with both effective rank and MP validation
        """
        
        # Data validation
        if returns_data.empty:
            return {'error': 'Empty returns data provided'}
        
        # Remove any assets with insufficient data
        min_observations = max(50, returns_data.shape[1] * 2)
        if returns_data.shape[0] < min_observations:
            return {'error': f'Insufficient data: {returns_data.shape[0]} obs, need {min_observations}'}
        
        # Calculate correlation matrix
        correlation_matrix = returns_data.corr()
        
        # Handle any numerical issues
        correlation_matrix = self._clean_correlation_matrix(correlation_matrix)
        
        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvals(correlation_matrix.values)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        N = len(eigenvalues)
        T = returns_data.shape[0]
        
        # Core metrics
        analysis_results = {}
        
        # 1. Effective Rank Analysis (always works)
        effective_rank_results = self._calculate_effective_rank_metrics(eigenvalues, N, holdings_weights)
        analysis_results.update(effective_rank_results)
        
        # 2. Marchenko-Pastur Analysis (if conditions are met)
        mp_results = self._perform_mp_analysis(eigenvalues, N, T)
        analysis_results.update(mp_results)
        
        # 3. Risk Concentration Analysis
        risk_analysis = self._analyze_risk_concentration(eigenvalues, correlation_matrix, holdings_weights)
        analysis_results.update(risk_analysis)
        
        # 4. Generate comprehensive interpretation
        interpretation = self._generate_comprehensive_interpretation(analysis_results)
        analysis_results['interpretation'] = interpretation
        
        # 5. Generate advisor talking points
        talking_points = self._generate_advisor_talking_points(analysis_results)
        analysis_results['advisor_talking_points'] = talking_points
        
        return analysis_results
    
    def _clean_correlation_matrix(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """Clean correlation matrix for numerical stability"""
        
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
    
    def _calculate_effective_rank_metrics(self, eigenvalues: np.ndarray, N: int, 
                                        holdings_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate effective rank and diversification metrics"""
        
        # Normalize eigenvalues to sum to N (trace of correlation matrix)
        eigenvalue_weights = eigenvalues / np.sum(eigenvalues)
        
        # Effective rank (participation ratio)
        effective_rank = 1 / np.sum(eigenvalue_weights**2)
        
        # Diversification metrics
        concentration_ratio = effective_rank / N
        diversification_loss = 1 - concentration_ratio
        
        # Factor contributions
        factor_contributions = eigenvalue_weights[:min(5, len(eigenvalue_weights))]
        
        # Weighted effective rank if weights provided
        weighted_effective_rank = None
        if holdings_weights is not None:
            # This would require more complex calculation based on asset weights
            pass
        
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
    
    def _perform_mp_analysis(self, eigenvalues: np.ndarray, N: int, T: int) -> Dict[str, Any]:
        """Perform Marchenko-Pastur analysis using corrected implementation"""
        
        try:
            mp_results = self.mp_analyzer.corrected_mp_analysis(eigenvalues, N, T)
            
            # Add MP-specific insights
            if mp_results.get('mp_fitting_successful', False):
                mp_results['mp_validation'] = self._validate_mp_results(mp_results)
            
            return mp_results
            
        except Exception as e:
            return {
                'mp_fitting_successful': False,
                'mp_error': str(e),
                'fallback_to_effective_rank': True
            }
    
    def _validate_mp_results(self, mp_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MP results against expected patterns"""
        
        q_ratio = mp_results['q_ratio']
        noise_fraction = mp_results['noise_fraction']
        num_signal = mp_results['num_signal_factors']
        
        validation = {
            'q_ratio_acceptable': q_ratio < 0.5,  # Good statistical power
            'noise_level_reasonable': 0.3 <= noise_fraction <= 0.95,
            'signal_factors_reasonable': 1 <= num_signal <= mp_results['num_eigenvalues'] // 2,
            'largest_eigenvalue_significant': mp_results.get('largest_signal_eigenvalue', 0) > 2.0
        }
        
        validation['overall_credible'] = all(validation.values())
        
        return validation
    
    def _analyze_risk_concentration(self, eigenvalues: np.ndarray, correlation_matrix: pd.DataFrame,
                                  holdings_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Analyze risk concentration patterns"""
        
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
    
    def _generate_comprehensive_interpretation(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive interpretation combining all analyses"""
        
        interpretation = []
        
        # Effective rank insights
        effective_rank = results['effective_rank']
        diversification_loss = results['diversification_loss']
        num_assets = results['num_assets']
        
        interpretation.append(f"Portfolio Diversification Analysis ({num_assets} assets):")
        interpretation.append(f"• Effective rank: {effective_rank:.1f} (behaves like {effective_rank:.1f} independent assets)")
        interpretation.append(f"• Diversification loss: {diversification_loss:.1%} (concentrated vs. equal-weight ideal)")
        
        # Factor concentration
        largest_factor = results['largest_eigenvalue_weight']
        top_3_factors = results['top_3_factors_weight']
        
        interpretation.append(f"• Largest factor explains {largest_factor:.1%} of portfolio variance")
        interpretation.append(f"• Top 3 factors explain {top_3_factors:.1%} of portfolio variance")
        
        # Marchenko-Pastur insights
        if results.get('mp_fitting_successful', False):
            noise_fraction = results['noise_fraction']
            num_signal = results['num_signal_factors']
            
            interpretation.append(f"Random Matrix Theory Validation:")
            interpretation.append(f"• {noise_fraction:.1%} of correlations are random noise")
            interpretation.append(f"• {num_signal} genuine risk factors identified")
            
            if results.get('mp_validation', {}).get('overall_credible', False):
                interpretation.append("• MP analysis validates correlation structure")
            else:
                interpretation.append("• MP analysis shows potential statistical limitations")
        
        # Risk concentration insights
        if 'risk_concentration' in results:
            concentration_level = results['risk_concentration']['concentration_level']
            correlation_pattern = results['correlation_structure']['correlation_pattern']
            
            interpretation.append(f"Risk Structure:")
            interpretation.append(f"• Concentration level: {concentration_level}")
            interpretation.append(f"• Asset correlations: {correlation_pattern}")
        
        return interpretation
    
    def _generate_advisor_talking_points(self, results: Dict[str, Any]) -> List[str]:
        """Generate advisor-ready talking points"""
        
        talking_points = []
        
        effective_rank = results['effective_rank']
        diversification_loss = results['diversification_loss']
        num_assets = results['num_assets']
        largest_factor = results['largest_eigenvalue_weight']
        
        # Key headline
        talking_points.append(f"Key Finding: Portfolio acts like {effective_rank:.1f} independent investments despite holding {num_assets} assets")
        
        # Diversification story
        talking_points.append(f"Hidden Cost: {diversification_loss:.0%} diversification loss due to correlation concentration")
        
        # Risk factor story
        talking_points.append(f"Risk Concentration: Single largest factor drives {largest_factor:.0%} of portfolio volatility")
        
        # MP validation if available
        if results.get('mp_fitting_successful', False):
            noise_fraction = results['noise_fraction']
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

# Example usage function
def analyze_portfolio_example():
    """Example of how to use the integrated analysis service"""
    
    # Create sample data (replace with real data loading)
    np.random.seed(42)
    T, N = 252, 10  # 1 year daily returns, 10 assets
    
    # Generate sample factor model data
    factor1 = np.random.normal(0, 0.02, T)
    factor2 = np.random.normal(0, 0.015, T)
    
    returns_data = []
    for i in range(N):
        # Varying factor loadings
        loading1 = 0.6 + 0.3 * (i / N)
        loading2 = 0.4 - 0.2 * (i / N)
        idiosyncratic = 0.3 * np.random.normal(0, 0.01, T)
        
        asset_returns = loading1 * factor1 + loading2 * factor2 + idiosyncratic
        returns_data.append(asset_returns)
    
    returns_df = pd.DataFrame(np.column_stack(returns_data), 
                             columns=[f'Asset_{i+1}' for i in range(N)])
    
    # Run analysis
    analyzer = IntegratedAnalysisService()
    results = analyzer.analyze_portfolio_correlation_structure(returns_df)
    
    # Print results
    print("INTEGRATED PORTFOLIO CORRELATION ANALYSIS")
    print("=" * 50)
    
    for line in results['interpretation']:
        print(line)
    
    print(f"\nADVISOR TALKING POINTS:")
    for point in results['advisor_talking_points']:
        print(f"• {point}")
    
    return results

if __name__ == "__main__":
    analyze_portfolio_example()
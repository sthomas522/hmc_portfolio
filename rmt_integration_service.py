import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings

# Import your existing improved analysis service as base
from improved_analysis_service import ImprovedAnalysisService

logger = logging.getLogger(__name__)

class RMTIntegratedAnalysisService(ImprovedAnalysisService):
    """
    Enhanced analysis service that integrates Random Matrix Theory
    Extends your existing ImprovedAnalysisService with RMT capabilities
    """
    
    def __init__(self, data_service):
        super().__init__(data_service)
        self.mp_enabled = True
        self.eigenportfolio_enabled = True
    
    def _marchenko_pastur_pdf(self, x: float, q: float, sigma2: float = 1.0) -> float:
        """
        Marchenko-Pastur probability density function
        
        Parameters:
        x: eigenvalue
        q: ratio N/T (variables/observations)
        sigma2: variance parameter
        """
        if q <= 0 or q > 1:
            return 0
        
        lambda_minus = sigma2 * (1 - np.sqrt(q))**2
        lambda_plus = sigma2 * (1 + np.sqrt(q))**2
        
        if x < lambda_minus or x > lambda_plus:
            return 0
        
        # MP density formula
        numerator = np.sqrt((lambda_plus - x) * (x - lambda_minus))
        denominator = 2 * np.pi * q * sigma2 * x
        
        return numerator / denominator if denominator > 0 else 0
    
    def _fit_marchenko_pastur_to_eigenvalues(self, eigenvalues: np.ndarray, 
                                           num_observations: int) -> Dict[str, Any]:
        """
        Fit Marchenko-Pastur distribution to eigenvalue spectrum
        This is the core of RMT noise detection
        """
        
        # Remove near-zero eigenvalues
        positive_eigenvals = eigenvalues[eigenvalues > 1e-8]
        N = len(positive_eigenvals)  # Number of assets
        T = num_observations  # Number of time observations
        q = N / T  # Critical ratio
        
        logger.info(f"MP fitting: N={N}, T={T}, q={q:.3f}")
        
        if q >= 1.0:
            logger.warning(f"q={q:.3f} >= 1: Not enough observations for reliable MP analysis")
            return self._fallback_mp_analysis(positive_eigenvals, q)
        
        if len(positive_eigenvals) < 3:
            logger.warning("Too few eigenvalues for MP fitting")
            return self._fallback_mp_analysis(positive_eigenvals, q)
        
        # Optimize sigma2 parameter to best fit the bulk of eigenvalues
        def mp_fitting_objective(sigma2):
            """
            Objective function: minimize difference between empirical and theoretical distributions
            """
            lambda_minus = sigma2 * (1 - np.sqrt(q))**2
            lambda_plus = sigma2 * (1 + np.sqrt(q))**2
            
            # Count eigenvalues within MP bounds (should be noise)
            within_bounds = np.sum((positive_eigenvals >= lambda_minus) & 
                                 (positive_eigenvals <= lambda_plus))
            
            # We expect most eigenvalues to be within bounds (noise)
            # Target: 80-95% of eigenvalues should be noise based on literature
            empirical_noise_fraction = within_bounds / len(positive_eigenvals)
            target_noise_fraction = 0.85  # Literature suggests ~85% noise
            
            # Minimize deviation from expected noise fraction
            error = abs(empirical_noise_fraction - target_noise_fraction)
            
            # Add penalty if bounds are unreasonable
            if lambda_minus < 0 or lambda_plus > 10:
                error += 10
            
            return error
        
        # Optimize sigma2
        try:
            result = minimize_scalar(mp_fitting_objective, 
                                   bounds=(0.1, 3.0), 
                                   method='bounded')
            sigma2_optimal = result.x
            fitting_successful = True
        except Exception as e:
            logger.warning(f"MP optimization failed: {e}")
            sigma2_optimal = 1.0
            fitting_successful = False
        
        # Calculate final classification with optimal parameters
        lambda_minus = sigma2_optimal * (1 - np.sqrt(q))**2
        lambda_plus = sigma2_optimal * (1 + np.sqrt(q))**2
        
        # Classify eigenvalues as signal or noise
        signal_mask = (positive_eigenvals < lambda_minus) | (positive_eigenvals > lambda_plus)
        noise_mask = ~signal_mask
        
        signal_eigenvalues = positive_eigenvals[signal_mask]
        noise_eigenvalues = positive_eigenvals[noise_mask]
        
        # Sort signal eigenvalues in descending order
        signal_eigenvalues = np.sort(signal_eigenvalues)[::-1]
        
        # Calculate key statistics
        noise_fraction = len(noise_eigenvalues) / len(positive_eigenvals)
        signal_variance_fraction = np.sum(signal_eigenvalues) / np.sum(positive_eigenvals)
        
        return {
            'mp_fitting_successful': fitting_successful,
            'q_ratio': float(q),
            'sigma2_optimal': float(sigma2_optimal),
            'lambda_minus': float(lambda_minus),
            'lambda_plus': float(lambda_plus),
            'num_eigenvalues': len(positive_eigenvals),
            'num_signal_factors': len(signal_eigenvalues),
            'num_noise_factors': len(noise_eigenvalues),
            'signal_eigenvalues': signal_eigenvalues.tolist(),
            'noise_eigenvalues': noise_eigenvalues.tolist(),
            'noise_fraction': float(noise_fraction),
            'signal_variance_fraction': float(signal_variance_fraction),
            'largest_signal_eigenvalue': float(signal_eigenvalues[0]) if len(signal_eigenvalues) > 0 else 0,
            'interpretation': self._interpret_mp_results(noise_fraction, len(signal_eigenvalues), q)
        }
    
    def _fallback_mp_analysis(self, eigenvalues: np.ndarray, q: float) -> Dict[str, Any]:
        """Fallback analysis when MP fitting fails"""
        
        return {
            'mp_fitting_successful': False,
            'q_ratio': float(q),
            'sigma2_optimal': 1.0,
            'lambda_minus': 0.0,
            'lambda_plus': 4.0,
            'num_eigenvalues': len(eigenvalues),
            'num_signal_factors': min(3, len(eigenvalues)),  # Conservative estimate
            'num_noise_factors': max(0, len(eigenvalues) - 3),
            'signal_eigenvalues': eigenvalues[:3].tolist() if len(eigenvalues) >= 3 else eigenvalues.tolist(),
            'noise_eigenvalues': eigenvalues[3:].tolist() if len(eigenvalues) > 3 else [],
            'noise_fraction': max(0, (len(eigenvalues) - 3) / len(eigenvalues)),
            'signal_variance_fraction': 0.5,
            'largest_signal_eigenvalue': float(eigenvalues[0]) if len(eigenvalues) > 0 else 0,
            'interpretation': ['MP fitting failed - using conservative estimates']
        }
    
    def _interpret_mp_results(self, noise_fraction: float, num_signal: int, q: float) -> List[str]:
        """Generate interpretation of MP analysis results"""
        
        interpretation = []
        
        # Noise level assessment
        if noise_fraction > 0.9:
            interpretation.append(f"🚨 EXTREME noise: {noise_fraction:.1%} of correlations are random")
            interpretation.append("Portfolio appears highly over-diversified with minimal genuine correlations")
        elif noise_fraction > 0.7:
            interpretation.append(f"⚠️ HIGH noise: {noise_fraction:.1%} of correlations are likely random")
            interpretation.append("Most apparent diversification is statistically indistinguishable from noise")
        elif noise_fraction > 0.5:
            interpretation.append(f"📊 MODERATE noise: {noise_fraction:.1%} of correlations are random")
        else:
            interpretation.append(f"✅ LOW noise: {noise_fraction:.1%} of correlations are random")
            interpretation.append("Portfolio shows genuine correlation structure")
        
        # Signal factor assessment
        if num_signal <= 2:
            interpretation.append(f"Dominated by {num_signal} major risk factor{'s' if num_signal > 1 else ''}")
        elif num_signal <= 5:
            interpretation.append(f"{num_signal} significant risk factors detected")
        else:
            interpretation.append(f"{num_signal} risk factors - reasonably diversified structure")
        
        # Data quality assessment
        if q > 0.8:
            interpretation.append(f"⚠️ Limited data: q={q:.2f} - results may be unreliable")
        elif q > 0.5:
            interpretation.append(f"Moderate data: q={q:.2f} - interpretable but limited")
        else:
            interpretation.append(f"Good data quality: q={q:.2f} - reliable RMT analysis")
        
        return interpretation
    
    def _analyze_eigenportfolios(self, correlation_matrix: pd.DataFrame,
                               eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                               tickers: List[str], max_portfolios: int = 5) -> Dict[str, Any]:
        """
        Analyze eigenportfolios to interpret economic meaning of factors
        """
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        eigenportfolios = []
        
        for i in range(min(max_portfolios, len(sorted_eigenvalues))):
            eigenvector = sorted_eigenvectors[:, i]
            eigenvalue = sorted_eigenvalues[i]
            
            # Calculate portfolio weights (normalize to unit leverage)
            raw_weights = eigenvector
            normalized_weights = raw_weights / np.sum(np.abs(raw_weights))
            
            # Find top holdings by absolute weight
            abs_weights = np.abs(normalized_weights)
            top_indices = np.argsort(abs_weights)[-10:][::-1]  # Top 10
            
            top_holdings = []
            for idx in top_indices:
                if abs_weights[idx] > 0.01:  # Only include meaningful weights
                    top_holdings.append({
                        'ticker': tickers[idx],
                        'weight': float(normalized_weights[idx]),
                        'abs_weight': float(abs_weights[idx])
                    })
            
            # Calculate portfolio statistics
            portfolio_stats = {
                'factor_number': i + 1,
                'eigenvalue': float(eigenvalue),
                'variance_explained': float(eigenvalue / np.sum(sorted_eigenvalues)),
                'cumulative_variance': float(np.sum(sorted_eigenvalues[:i+1]) / np.sum(sorted_eigenvalues)),
                'top_holdings': top_holdings,
                'weight_concentration': float(np.sum(normalized_weights**2)),  # Herfindahl index
                'max_abs_weight': float(np.max(abs_weights)),
                'num_significant_holdings': int(np.sum(abs_weights > 0.05)),
                'long_short_ratio': float(np.sum(normalized_weights > 0) / len(normalized_weights)),
                'interpretation': self._interpret_eigenportfolio(normalized_weights, tickers, i, abs_weights)
            }
            
            eigenportfolios.append(portfolio_stats)
        
        return {
            'eigenportfolios': eigenportfolios,
            'total_variance_top3': float(np.sum(sorted_eigenvalues[:3]) / np.sum(sorted_eigenvalues)),
            'total_variance_top5': float(np.sum(sorted_eigenvalues[:5]) / np.sum(sorted_eigenvalues)),
            'eigenvalue_concentration': float(sorted_eigenvalues[0] / np.sum(sorted_eigenvalues)) if len(sorted_eigenvalues) > 0 else 0
        }
    
    def _interpret_eigenportfolio(self, weights: np.ndarray, tickers: List[str], 
                                factor_num: int, abs_weights: np.ndarray) -> str:
        """
        Provide economic interpretation of eigenportfolios
        """
        
        # Find dominant holdings
        top_indices = np.argsort(abs_weights)[-5:][::-1]
        top_tickers = [tickers[i] for i in top_indices if abs_weights[i] > 0.05]
        
        # Analyze signs (long vs short)
        positive_weights = weights > 0.05
        negative_weights = weights < -0.05
        num_positive = np.sum(positive_weights)
        num_negative = np.sum(negative_weights)
        
        interpretation = f"Factor {factor_num + 1}: "
        
        if factor_num == 0:
            # First factor: usually market factor
            if num_positive > len(weights) * 0.8 and num_negative < len(weights) * 0.1:
                interpretation += "Market factor (broad positive exposure)"
            else:
                interpretation += f"Dominant factor - concentrated in {', '.join(top_tickers[:3])}"
        
        elif factor_num == 1:
            # Second factor: often sector or style
            if len(top_tickers) >= 2:
                interpretation += f"Secondary factor - {', '.join(top_tickers[:3])}"
                # Try to identify sector patterns
                if any('NVDA' in t or 'AMD' in t or 'AVGO' in t for t in top_tickers):
                    interpretation += " (semiconductor/tech focus)"
                elif any('JPM' in t or 'BAC' in t or 'WFC' in t for t in top_tickers):
                    interpretation += " (financials focus)"
            else:
                interpretation += "Secondary factor"
        
        else:
            # Higher order factors
            interpretation += f"Factor {factor_num + 1}"
            if len(top_tickers) >= 2:
                interpretation += f" - {', '.join(top_tickers[:2])}"
        
        # Add concentration assessment
        concentration = np.sum(abs_weights**2)
        if concentration > 0.15:
            interpretation += " (highly concentrated)"
        elif concentration < 0.03:
            interpretation += " (broadly diversified)"
        
        # Add long/short information
        if num_negative > len(weights) * 0.2:
            interpretation += " (long-short structure)"
        
        return interpretation
    
    def _robust_correlation_analysis(self, returns: pd.DataFrame, 
                                   weights: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced correlation analysis with RMT integration
        Extends the parent class method with MP and eigenportfolio analysis
        """
        
        # Call parent method for base analysis
        base_analysis = super()._robust_correlation_analysis(returns, weights)
        
        # Add RMT enhancements if enabled
        if self.mp_enabled or self.eigenportfolio_enabled:
            correlation_matrix = returns.corr().fillna(0)
            np.fill_diagonal(correlation_matrix.values, 1.0)
            
            # Get eigenvalues and eigenvectors for RMT analysis
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix.values)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = eigenvalues[eigenvalues > 1e-12]
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                rmt_analysis_possible = True
                
            except Exception as e:
                logger.error(f"Eigendecomposition failed for RMT analysis: {e}")
                eigenvalues = np.array([1.0])
                eigenvectors = np.eye(len(returns.columns))
                rmt_analysis_possible = False
            
            # Marchenko-Pastur analysis
            if self.mp_enabled and rmt_analysis_possible:
                mp_analysis = self._fit_marchenko_pastur_to_eigenvalues(
                    eigenvalues, len(returns)
                )
                base_analysis['marchenko_pastur_analysis'] = mp_analysis
            
            # Eigenportfolio analysis
            if self.eigenportfolio_enabled and rmt_analysis_possible:
                eigenportfolio_analysis = self._analyze_eigenportfolios(
                    correlation_matrix, eigenvalues, eigenvectors, 
                    returns.columns.tolist()
                )
                base_analysis['eigenportfolio_analysis'] = eigenportfolio_analysis
            
            # Generate integrated insights
            base_analysis['rmt_insights'] = self._generate_integrated_insights(
                base_analysis
            )
        
        return base_analysis
    
    def _generate_integrated_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate insights that integrate traditional and RMT analysis
        """
        
        insights = []
        
        # Basic metrics
        effective_rank = analysis_results.get('effective_rank', 0)
        concentration_ratio = analysis_results.get('concentration_ratio', 0)
        
        insights.append(f"📊 Portfolio Analysis Summary:")
        insights.append(f"• Effective Rank: {effective_rank:.2f}")
        insights.append(f"• Diversification Efficiency: {concentration_ratio:.1%}")
        
        # Marchenko-Pastur insights
        if 'marchenko_pastur_analysis' in analysis_results:
            mp = analysis_results['marchenko_pastur_analysis']
            if mp['mp_fitting_successful']:
                insights.append(f"\n🔬 Random Matrix Theory Analysis:")
                insights.extend([f"• {insight}" for insight in mp['interpretation']])
                
                # Compare effective rank to RMT signal factors
                rmt_factors = mp['num_signal_factors']
                if abs(effective_rank - rmt_factors) < 1:
                    insights.append(f"• ✅ Effective rank aligns with RMT signal factors ({rmt_factors})")
                else:
                    insights.append(f"• 🔍 Effective rank ({effective_rank:.1f}) vs RMT factors ({rmt_factors})")
        
        # Eigenportfolio insights
        if 'eigenportfolio_analysis' in analysis_results:
            ep = analysis_results['eigenportfolio_analysis']
            if ep['eigenportfolios']:
                insights.append(f"\n🎯 Factor Structure:")
                
                # First factor dominance
                first_factor = ep['eigenportfolios'][0]
                variance_explained = first_factor['variance_explained']
                insights.append(f"• {first_factor['interpretation']}")
                insights.append(f"• Explains {variance_explained:.1%} of portfolio variance")
                
                if variance_explained > 0.6:
                    insights.append(f"• ⚠️ Extreme factor concentration - high single-factor risk")
                elif variance_explained > 0.4:
                    insights.append(f"• ⚠️ High factor concentration")
                
                # Top 3 factor concentration
                top3_variance = ep['total_variance_top3']
                insights.append(f"• Top 3 factors explain {top3_variance:.1%} of variance")
        
        return insights

# Test function to validate the implementation
async def test_rmt_integration():
    """Test the RMT integration with a sample portfolio"""
    
    print("🧪 Testing RMT Integration")
    print("=" * 50)
    
    # Import your data service
    import sys
    from pathlib import Path
    backend_path = Path(__file__).parent / 'backend'
    sys.path.append(str(backend_path))
    
    from app.services.data_service import DataService
    
    # Test portfolio - the VTIVX top 10 we analyzed before
    test_tickers = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'JPM']
    
    try:
        # Initialize services
        ds = DataService()
        rmt_service = RMTIntegratedAnalysisService(ds)
        
        print(f"🔬 Analyzing {len(test_tickers)} tickers with RMT enhancements...")
        
        # Run enhanced analysis
        result = await rmt_service.analyze_portfolio(
            test_tickers, 
            analysis_types=['correlation']
        )
        
        # Display results
        print(f"\n📊 Enhanced Analysis Results:")
        
        # Basic metrics
        correlation = result['correlation']
        print(f"Effective Rank: {correlation['effective_rank']:.2f}")
        print(f"Concentration Ratio: {correlation['concentration_ratio']:.1%}")
        
        # Marchenko-Pastur results
        if 'marchenko_pastur_analysis' in correlation:
            mp = correlation['marchenko_pastur_analysis']
            print(f"\n🔬 Marchenko-Pastur Analysis:")
            print(f"Q-ratio: {mp['q_ratio']:.3f}")
            print(f"Signal factors: {mp['num_signal_factors']}")
            print(f"Noise fraction: {mp['noise_fraction']:.1%}")
            print(f"Largest signal eigenvalue: {mp['largest_signal_eigenvalue']:.2f}")
        
        # Eigenportfolio results
        if 'eigenportfolio_analysis' in correlation:
            ep = correlation['eigenportfolio_analysis']
            print(f"\n🎯 Eigenportfolio Analysis:")
            
            for i, portfolio in enumerate(ep['eigenportfolios'][:3]):
                print(f"\nFactor {i+1}:")
                print(f"  {portfolio['interpretation']}")
                print(f"  Variance explained: {portfolio['variance_explained']:.1%}")
                print(f"  Top holdings: {[h['ticker'] for h in portfolio['top_holdings'][:3]]}")
        
        # Integrated insights
        if 'rmt_insights' in correlation:
            print(f"\n💡 Integrated Insights:")
            for insight in correlation['rmt_insights']:
                print(f"  {insight}")
        
        print(f"\n✅ RMT Integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ RMT Integration test failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rmt_integration())
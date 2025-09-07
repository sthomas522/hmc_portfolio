import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
import warnings

# Import the corrected MP implementation and base service
from fixed_mp_implementation import FixedMarchenkoPassturAnalysis
from improved_analysis_service import ImprovedAnalysisService

logger = logging.getLogger(__name__)

class CorrectedRMTAnalysisService(ImprovedAnalysisService):
    """
    Corrected RMT analysis service with fixed Marchenko-Pastur implementation
    """
    
    def __init__(self, data_service):
        super().__init__(data_service)
        self.mp_analyzer = FixedMarchenkoPassturAnalysis()
        self.mp_enabled = True
        self.eigenportfolio_enabled = True
    
    def _robust_correlation_analysis(self, returns: pd.DataFrame, 
                                   weights: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced correlation analysis with CORRECTED RMT implementation
        """
        
        # Call parent method for base analysis
        base_analysis = super()._robust_correlation_analysis(returns, weights)
        
        # Add corrected RMT enhancements
        if self.mp_enabled or self.eigenportfolio_enabled:
            correlation_matrix = returns.corr().fillna(0)
            np.fill_diagonal(correlation_matrix.values, 1.0)
            
            # Get eigenvalues and eigenvectors for RMT analysis
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix.values)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = eigenvalues[eigenvalues > 1e-12]
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                N = len(returns.columns)  # Number of assets
                T = len(returns)  # Number of observations
                
                rmt_analysis_possible = True
                
            except Exception as e:
                logger.error(f"Eigendecomposition failed for RMT analysis: {e}")
                eigenvalues = np.array([1.0])
                eigenvectors = np.eye(len(returns.columns))
                N = len(returns.columns)
                T = len(returns)
                rmt_analysis_possible = False
            
            # CORRECTED Marchenko-Pastur analysis
            if self.mp_enabled and rmt_analysis_possible:
                logger.info(f"Running corrected MP analysis: N={N}, T={T}")
                mp_analysis = self.mp_analyzer.fit_marchenko_pastur_corrected(
                    eigenvalues, N, T
                )
                base_analysis['marchenko_pastur_analysis'] = mp_analysis
                
                # Add comparison with effective rank
                effective_rank = base_analysis.get('effective_rank', 0)
                mp_signal_factors = mp_analysis['num_signal_factors']
                
                base_analysis['rmt_validation'] = {
                    'effective_rank': effective_rank,
                    'mp_signal_factors': mp_signal_factors,
                    'rank_vs_signal_ratio': effective_rank / mp_signal_factors if mp_signal_factors > 0 else 0,
                    'consistency_check': self._check_rmt_consistency(effective_rank, mp_signal_factors),
                }
            
            # Enhanced eigenportfolio analysis
            if self.eigenportfolio_enabled and rmt_analysis_possible:
                eigenportfolio_analysis = self._analyze_eigenportfolios_enhanced(
                    correlation_matrix, eigenvalues, eigenvectors, 
                    returns.columns.tolist(), mp_analysis if self.mp_enabled else None
                )
                base_analysis['eigenportfolio_analysis'] = eigenportfolio_analysis
            
            # Generate enhanced integrated insights
            base_analysis['rmt_insights'] = self._generate_corrected_insights(
                base_analysis, N, T
            )
        
        return base_analysis
    
    def _check_rmt_consistency(self, effective_rank: float, mp_signal_factors: int) -> str:
        """Check consistency between effective rank and MP signal factors"""
        
        if mp_signal_factors == 0:
            return "⚠️ MP detects no signal factors - possible pure noise portfolio"
        
        ratio = effective_rank / mp_signal_factors
        
        if 0.8 <= ratio <= 1.5:
            return "✅ Effective rank aligns well with MP signal factors"
        elif 0.5 <= ratio < 0.8:
            return "🔍 Effective rank lower than MP signals - high correlation within factors"
        elif 1.5 < ratio <= 2.5:
            return "🔍 Effective rank higher than MP signals - possible intermediate factors"
        else:
            return "⚠️ Significant discrepancy between effective rank and MP analysis"
    
    def _analyze_eigenportfolios_enhanced(self, correlation_matrix: pd.DataFrame,
                                        eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                                        tickers: List[str], mp_analysis: Optional[Dict] = None,
                                        max_portfolios: int = 5) -> Dict[str, Any]:
        """
        Enhanced eigenportfolio analysis with MP signal/noise classification
        """
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        eigenportfolios = []
        
        # Use MP analysis to identify signal factors if available
        signal_factors = set()
        if mp_analysis and mp_analysis['mp_fitting_successful']:
            signal_eigenvals = mp_analysis['signal_eigenvalues']
            # Match signal eigenvalues to sorted eigenvalues
            for i, eigenval in enumerate(sorted_eigenvalues):
                if any(abs(eigenval - sig_val) < 1e-6 for sig_val in signal_eigenvals):
                    signal_factors.add(i)
        
        for i in range(min(max_portfolios, len(sorted_eigenvalues))):
            eigenvector = sorted_eigenvectors[:, i]
            eigenvalue = sorted_eigenvalues[i]
            
            # Determine if this is a signal or noise factor
            is_signal_factor = i in signal_factors or mp_analysis is None
            
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
            
            # Enhanced interpretation with MP context
            interpretation = self._interpret_eigenportfolio_enhanced(
                normalized_weights, tickers, i, abs_weights, is_signal_factor
            )
            
            # Calculate portfolio statistics
            portfolio_stats = {
                'factor_number': i + 1,
                'eigenvalue': float(eigenvalue),
                'variance_explained': float(eigenvalue / np.sum(sorted_eigenvalues)),
                'cumulative_variance': float(np.sum(sorted_eigenvalues[:i+1]) / np.sum(sorted_eigenvalues)),
                'is_signal_factor': is_signal_factor,
                'top_holdings': top_holdings,
                'weight_concentration': float(np.sum(normalized_weights**2)),  # Herfindahl index
                'max_abs_weight': float(np.max(abs_weights)),
                'num_significant_holdings': int(np.sum(abs_weights > 0.05)),
                'long_short_ratio': float(np.sum(normalized_weights > 0) / len(normalized_weights)),
                'interpretation': interpretation
            }
            
            eigenportfolios.append(portfolio_stats)
        
        # Calculate signal vs noise factor statistics
        signal_portfolios = [ep for ep in eigenportfolios if ep['is_signal_factor']]
        noise_portfolios = [ep for ep in eigenportfolios if not ep['is_signal_factor']]
        
        signal_variance = sum(ep['variance_explained'] for ep in signal_portfolios)
        noise_variance = sum(ep['variance_explained'] for ep in noise_portfolios)
        
        return {
            'eigenportfolios': eigenportfolios,
            'total_variance_top3': float(np.sum(sorted_eigenvalues[:3]) / np.sum(sorted_eigenvalues)),
            'total_variance_top5': float(np.sum(sorted_eigenvalues[:5]) / np.sum(sorted_eigenvalues)),
            'eigenvalue_concentration': float(sorted_eigenvalues[0] / np.sum(sorted_eigenvalues)) if len(sorted_eigenvalues) > 0 else 0,
            'signal_factor_stats': {
                'num_signal_factors': len(signal_portfolios),
                'signal_variance_explained': float(signal_variance),
                'avg_signal_concentration': float(np.mean([ep['weight_concentration'] for ep in signal_portfolios])) if signal_portfolios else 0
            },
            'noise_factor_stats': {
                'num_noise_factors': len(noise_portfolios),
                'noise_variance_explained': float(noise_variance),
                'avg_noise_concentration': float(np.mean([ep['weight_concentration'] for ep in noise_portfolios])) if noise_portfolios else 0
            }
        }
    
    def _interpret_eigenportfolio_enhanced(self, weights: np.ndarray, tickers: List[str], 
                                         factor_num: int, abs_weights: np.ndarray,
                                         is_signal_factor: bool) -> str:
        """
        Enhanced eigenportfolio interpretation with MP signal/noise context
        """
        
        # Find dominant holdings
        top_indices = np.argsort(abs_weights)[-5:][::-1]
        top_tickers = [tickers[i] for i in top_indices if abs_weights[i] > 0.05]
        
        # Analyze signs (long vs short)
        positive_weights = weights > 0.05
        negative_weights = weights < -0.05
        num_positive = np.sum(positive_weights)
        num_negative = np.sum(negative_weights)
        
        # Base interpretation
        if is_signal_factor:
            if factor_num == 0:
                base_interpretation = "Market factor"
            elif factor_num == 1:
                base_interpretation = "Secondary factor"
            else:
                base_interpretation = f"Signal factor {factor_num + 1}"
        else:
            base_interpretation = f"Noise factor {factor_num + 1}"
        
        # Enhanced semantic interpretation
        if len(top_tickers) >= 2:
            # Check for specific patterns
            if 'GOOGL' in top_tickers and 'GOOG' in top_tickers:
                semantic = "Alphabet dual-class structure"
            elif any(t in ['NVDA', 'AMD', 'AVGO', 'QCOM', 'MRVL'] for t in top_tickers):
                if sum(1 for t in top_tickers if t in ['NVDA', 'AMD', 'AVGO', 'QCOM', 'MRVL']) >= 2:
                    semantic = "Semiconductor sector exposure"
                else:
                    semantic = f"Tech concentration ({', '.join(top_tickers[:3])})"
            elif any(t in ['JPM', 'BAC', 'WFC', 'GS', 'MS'] for t in top_tickers):
                if sum(1 for t in top_tickers if t in ['JPM', 'BAC', 'WFC', 'GS', 'MS']) >= 2:
                    semantic = "Financial sector exposure"
                else:
                    semantic = f"Mixed exposure ({', '.join(top_tickers[:3])})"
            else:
                semantic = f"Concentrated in {', '.join(top_tickers[:3])}"
        else:
            semantic = "Broadly diversified"
        
        interpretation = f"{base_interpretation}: {semantic}"
        
        # Add concentration assessment
        concentration = np.sum(abs_weights**2)
        if concentration > 0.15:
            interpretation += " (highly concentrated)"
        elif concentration < 0.03:
            interpretation += " (broadly diversified)"
        
        # Add long/short information
        if num_negative > len(weights) * 0.2:
            interpretation += " [long-short]"
        
        return interpretation
    
    def _generate_corrected_insights(self, analysis_results: Dict[str, Any], 
                                   N: int, T: int) -> List[str]:
        """
        Generate insights with corrected RMT analysis
        """
        
        insights = []
        
        # Basic metrics
        effective_rank = analysis_results.get('effective_rank', 0)
        concentration_ratio = analysis_results.get('concentration_ratio', 0)
        
        insights.append(f"Portfolio Analysis Summary:")
        insights.append(f"• Holdings: {N} assets, {T} observations")
        insights.append(f"• Effective Rank: {effective_rank:.2f}")
        insights.append(f"• Diversification Efficiency: {concentration_ratio:.1%}")
        
        # Enhanced MP insights
        if 'marchenko_pastur_analysis' in analysis_results:
            mp = analysis_results['marchenko_pastur_analysis']
            
            insights.append(f"\nRandom Matrix Theory Analysis:")
            insights.append(f"• Q-ratio: {mp['q_ratio']:.3f} (assets/observations)")
            
            for interpretation in mp['interpretation']:
                insights.append(f"• {interpretation}")
            
            # RMT vs Effective Rank comparison
            if 'rmt_validation' in analysis_results:
                validation = analysis_results['rmt_validation']
                insights.append(f"\nRMT Validation:")
                insights.append(f"• Effective Rank: {validation['effective_rank']:.1f}")
                insights.append(f"• MP Signal Factors: {validation['mp_signal_factors']}")
                insights.append(f"• {validation['consistency_check']}")
        
        # Enhanced eigenportfolio insights
        if 'eigenportfolio_analysis' in analysis_results:
            ep = analysis_results['eigenportfolio_analysis']
            if ep['eigenportfolios']:
                insights.append(f"\nFactor Structure Analysis:")
                
                # Focus on signal factors
                signal_stats = ep['signal_factor_stats']
                insights.append(f"• Signal factors: {signal_stats['num_signal_factors']}")
                insights.append(f"• Signal variance: {signal_stats['signal_variance_explained']:.1%}")
                
                # Show top factors with enhanced interpretation
                for i, portfolio in enumerate(ep['eigenportfolios'][:3]):
                    factor_type = "Signal" if portfolio['is_signal_factor'] else "Noise"
                    insights.append(f"• {portfolio['interpretation']} ({portfolio['variance_explained']:.1%} variance)")
        
        return insights

# Test function for the corrected implementation
async def test_corrected_rmt_on_vtivx():
    """Test corrected RMT implementation on VTIVX data"""
    
    print("Testing Corrected RMT Implementation on VTIVX")
    print("=" * 60)
    
    # Import data service
    import sys
    from pathlib import Path
    backend_path = Path(__file__).parent / 'backend'
    sys.path.append(str(backend_path))
    
    from app.services.data_service import DataService
    
    # VTIVX top holdings
    vtivx_holdings = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'JPM']
    
    try:
        # Initialize corrected service
        ds = DataService()
        corrected_service = CorrectedRMTAnalysisService(ds)
        
        print(f"Analyzing {len(vtivx_holdings)} VTIVX holdings with corrected RMT...")
        
        # Run analysis
        result = await corrected_service.analyze_portfolio(
            vtivx_holdings, 
            analysis_types=['correlation']
        )
        
        # Display results
        correlation = result['correlation']
        metadata = result['metadata']
        
        print(f"\nData Quality: {metadata['success_rate']:.1%} success")
        print(f"Effective Rank: {correlation['effective_rank']:.2f}")
        print(f"Concentration Ratio: {correlation['concentration_ratio']:.1%}")
        
        # Show corrected MP analysis
        if 'marchenko_pastur_analysis' in correlation:
            mp = correlation['marchenko_pastur_analysis']
            
            print(f"\nCorrected Marchenko-Pastur Analysis:")
            print(f"Q-ratio: {mp['q_ratio']:.4f}")
            print(f"Signal factors: {mp['num_signal_factors']}")
            print(f"Noise fraction: {mp['noise_fraction']:.1%}")
            print(f"Fitting successful: {mp['mp_fitting_successful']}")
            
            print(f"\nMP Interpretation:")
            for interpretation in mp['interpretation']:
                print(f"  • {interpretation}")
        
        # Show RMT validation
        if 'rmt_validation' in correlation:
            validation = correlation['rmt_validation']
            print(f"\nRMT Validation:")
            print(f"Effective Rank vs MP Signals: {validation['effective_rank']:.1f} vs {validation['mp_signal_factors']}")
            print(f"Consistency: {validation['consistency_check']}")
        
        # Show enhanced eigenportfolios
        if 'eigenportfolio_analysis' in correlation:
            ep = correlation['eigenportfolio_analysis']
            print(f"\nEnhanced Factor Analysis:")
            
            for i, portfolio in enumerate(ep['eigenportfolios'][:3]):
                factor_type = "Signal" if portfolio['is_signal_factor'] else "Noise"
                print(f"Factor {i+1} ({factor_type}): {portfolio['interpretation']}")
                print(f"  Variance: {portfolio['variance_explained']:.1%}")
        
        # Show corrected insights
        if 'rmt_insights' in correlation:
            print(f"\nCorrected RMT Insights:")
            for insight in correlation['rmt_insights']:
                print(f"  {insight}")
        
        print(f"\nCorrected RMT analysis complete!")
        return True
        
    except Exception as e:
        print(f"Corrected RMT test failed: {e}")
        return False

# Run validation test
def run_mp_validation():
    """Run MP implementation validation"""
    
    print("Validating Corrected MP Implementation")
    print("=" * 50)
    
    mp_analyzer = FixedMarchenkoPassturAnalysis()
    validation_success = mp_analyzer.validate_implementation()
    
    if validation_success:
        print("\nRunning realistic portfolio test...")
        realistic_test = test_corrected_mp_implementation()
        return realistic_test
    else:
        print("\nValidation failed - implementation has issues")
        return False

if __name__ == "__main__":
    import asyncio
    
    # First validate the MP implementation
    validation_success = run_mp_validation()
    
    if validation_success:
        print("\n" + "=" * 60)
        # Then test on VTIVX
        asyncio.run(test_corrected_rmt_on_vtivx())
    else:
        print("Fix MP implementation before testing on VTIVX")
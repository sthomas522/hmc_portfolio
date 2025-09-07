import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class CorrectedMPImplementation:
    """
    MP implementation with corrected sigma² selection
    """
    
    def __init__(self):
        pass
    
    def adaptive_sigma2_selection(self, eigenvalues: np.ndarray, q: float) -> float:
        """
        Adaptive sigma² selection based on eigenvalue distribution analysis
        """
        
        # Sort eigenvalues descending
        sorted_eigenvals = np.sort(eigenvalues)[::-1]
        
        print(f"\nADAPTIVE σ² SELECTION:")
        print(f"Eigenvalues: {sorted_eigenvals}")
        
        # Method 1: Identify the noise floor
        # For factor models, small eigenvalues represent the noise level
        # These should be INSIDE the MP bounds (classified as noise)
        
        # Find natural breaks in the eigenvalue spectrum
        eigenval_ratios = sorted_eigenvals[:-1] / sorted_eigenvals[1:]
        large_jumps = eigenval_ratios > 3.0  # Factor of 3+ indicates signal vs noise
        
        if np.any(large_jumps):
            # Find the first large jump - this separates signal from noise
            first_jump = np.where(large_jumps)[0][0]
            signal_eigenvals = sorted_eigenvals[:first_jump+1]
            noise_eigenvals = sorted_eigenvals[first_jump+1:]
            
            print(f"Natural break detected at position {first_jump}")
            print(f"Signal eigenvalues: {signal_eigenvals}")
            print(f"Noise eigenvalues: {noise_eigenvals}")
            
            if len(noise_eigenvals) > 0:
                # Scale sigma² so that noise eigenvalues fall within MP bounds
                noise_level = np.mean(noise_eigenvals)
                
                # For MP distribution, the center is approximately at sigma²
                # Scale sigma² so noise eigenvalues are near the center
                estimated_sigma2 = noise_level / 1.0  # Rough scaling
                
                print(f"Noise level: {noise_level:.4f}")
                print(f"Estimated σ² from noise: {estimated_sigma2:.4f}")
                
                return estimated_sigma2
        
        # Method 2: Use median of smaller eigenvalues
        # If no clear break, use the smaller half of eigenvalues as noise reference
        n_half = len(sorted_eigenvals) // 2
        smaller_half = sorted_eigenvals[n_half:]
        median_small = np.median(smaller_half)
        
        print(f"No clear break found")
        print(f"Smaller half eigenvalues: {smaller_half}")
        print(f"Median of smaller half: {median_small:.4f}")
        
        # Scale sigma² based on the noise level
        estimated_sigma2 = median_small / 0.5  # Scale factor to put noise in middle of bounds
        
        print(f"Estimated σ² from median: {estimated_sigma2:.4f}")
        
        return max(0.1, min(3.0, estimated_sigma2))  # Reasonable bounds
    
    def corrected_mp_analysis(self, eigenvalues: np.ndarray, N: int, T: int) -> Dict[str, Any]:
        """
        MP analysis with corrected sigma² selection
        """
        
        if len(eigenvalues) == 0:
            return {'mp_fitting_successful': False, 'reason': 'No eigenvalues'}
        
        positive_eigenvals = eigenvalues[eigenvalues > 1e-10]
        positive_eigenvals = np.sort(positive_eigenvals)[::-1]
        
        q = N / T
        
        if q >= 1.0:
            return {'mp_fitting_successful': False, 'reason': f'q >= 1 (q={q:.3f})'}
        
        print(f"\n=== CORRECTED MP ANALYSIS ===")
        print(f"N={N}, T={T}, q={q:.4f}")
        
        # Use adaptive sigma² selection
        optimal_sigma2 = self.adaptive_sigma2_selection(positive_eigenvals, q)
        
        # Calculate bounds with optimal sigma²
        lambda_minus = optimal_sigma2 * (1 - np.sqrt(q))**2
        lambda_plus = optimal_sigma2 * (1 + np.sqrt(q))**2
        
        print(f"\nFinal classification with σ²={optimal_sigma2:.4f}:")
        print(f"Bounds: [{lambda_minus:.4f}, {lambda_plus:.4f}]")
        
        # Classification: outside bounds = signal, inside bounds = noise
        signal_mask = (positive_eigenvals < lambda_minus) | (positive_eigenvals > lambda_plus)
        noise_mask = ~signal_mask
        
        signal_eigenvalues = positive_eigenvals[signal_mask]
        noise_eigenvalues = positive_eigenvals[noise_mask]
        
        print(f"Signal eigenvalues: {signal_eigenvalues}")
        print(f"Noise eigenvalues: {noise_eigenvalues}")
        
        # Calculate statistics
        noise_fraction = len(noise_eigenvalues) / len(positive_eigenvals)
        signal_variance_fraction = np.sum(signal_eigenvalues) / np.sum(positive_eigenvals)
        
        print(f"Noise fraction: {noise_fraction:.1%}")
        print(f"Signal factors: {len(signal_eigenvalues)}")
        
        # Interpretation
        interpretation = self._generate_interpretation(noise_fraction, len(signal_eigenvalues), q)
        
        return {
            'mp_fitting_successful': True,
            'q_ratio': float(q),
            'sigma2_optimal': float(optimal_sigma2),
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
            'interpretation': interpretation
        }
    
    def _generate_interpretation(self, noise_fraction: float, num_signal: int, q: float) -> List[str]:
        """Generate interpretation of corrected results"""
        
        interpretation = []
        
        # Data quality
        if q > 0.2:
            interpretation.append(f"⚠️ High q-ratio ({q:.2f}) - limited observations")
        else:
            interpretation.append(f"✅ Good q-ratio ({q:.2f}) - reliable analysis")
        
        # Noise assessment
        if noise_fraction > 0.8:
            interpretation.append(f"🔬 VERY HIGH noise: {noise_fraction:.1%} of correlations are random")
        elif noise_fraction > 0.6:
            interpretation.append(f"🔬 HIGH noise: {noise_fraction:.1%} of correlations are random")
        elif noise_fraction > 0.4:
            interpretation.append(f"🔬 MODERATE noise: {noise_fraction:.1%} of correlations are random")
        elif noise_fraction > 0.2:
            interpretation.append(f"🔬 LOW noise: {noise_fraction:.1%} of correlations are random")
        else:
            interpretation.append(f"🔬 MINIMAL noise: {noise_fraction:.1%} of correlations are random")
        
        # Signal factors
        if num_signal <= 2:
            interpretation.append(f"📊 {num_signal} dominant risk factors - highly concentrated")
        elif num_signal <= 4:
            interpretation.append(f"📊 {num_signal} risk factors - moderately concentrated")
        else:
            interpretation.append(f"📊 {num_signal} risk factors - well-diversified")
        
        return interpretation
    
    def test_corrected_implementation(self):
        """Test the corrected implementation on all three cases"""
        
        print("TESTING CORRECTED MP IMPLEMENTATION")
        print("=" * 50)
        
        # Test Case 1: Random data
        print("\nTEST 1: Random Uncorrelated Data")
        np.random.seed(42)
        N, T = 10, 1000
        random_data = np.random.normal(0, 1, (T, N))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvals = np.linalg.eigvals(random_corr)
        
        random_result = self.corrected_mp_analysis(random_eigenvals, N, T)
        
        print(f"\nResults:")
        print(f"  Noise fraction: {random_result['noise_fraction']:.1%}")
        print(f"  Signal factors: {random_result['num_signal_factors']}")
        
        # Test Case 2: Single factor
        print("\n" + "="*60)
        print("\nTEST 2: Single Factor Model")
        factor = np.random.normal(0, 1, T)
        single_factor_data = []
        for i in range(N):
            loading = 0.8
            noise = 0.2 * np.random.normal(0, 1, T)
            asset_return = loading * factor + noise
            single_factor_data.append(asset_return)
        
        single_factor_data = np.column_stack(single_factor_data)
        single_corr = np.corrcoef(single_factor_data, rowvar=False)
        single_eigenvals = np.linalg.eigvals(single_corr)
        
        single_result = self.corrected_mp_analysis(single_eigenvals, N, T)
        
        print(f"\nResults:")
        print(f"  Noise fraction: {single_result['noise_fraction']:.1%}")
        print(f"  Signal factors: {single_result['num_signal_factors']}")
        
        # Test Case 3: Two factors
        print("\n" + "="*60)
        print("\nTEST 3: Two Factor Model")
        factor1 = np.random.normal(0, 1, T)
        factor2 = np.random.normal(0, 1, T)
        two_factor_data = []
        for i in range(N):
            if i < N//2:
                loading1, loading2 = 0.7, 0.3
            else:
                loading1, loading2 = 0.3, 0.7
            noise = 0.3 * np.random.normal(0, 1, T)
            asset_return = loading1 * factor1 + loading2 * factor2 + noise
            two_factor_data.append(asset_return)
        
        two_factor_data = np.column_stack(two_factor_data)
        two_corr = np.corrcoef(two_factor_data, rowvar=False)
        two_eigenvals = np.linalg.eigvals(two_corr)
        
        two_result = self.corrected_mp_analysis(two_eigenvals, N, T)
        
        print(f"\nResults:")
        print(f"  Noise fraction: {two_result['noise_fraction']:.1%}")
        print(f"  Signal factors: {two_result['num_signal_factors']}")
        
        # Validation
        print(f"\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        
        # Expected results
        expected_results = {
            'random': {'noise_range': (0.7, 0.95), 'signal_range': (0, 2)},
            'single': {'noise_range': (0.7, 0.95), 'signal_range': (1, 2)},
            'two': {'noise_range': (0.6, 0.9), 'signal_range': (2, 3)}
        }
        
        results = [
            ('Random', random_result, expected_results['random']),
            ('Single Factor', single_result, expected_results['single']),
            ('Two Factor', two_result, expected_results['two'])
        ]
        
        all_passed = True
        
        for name, result, expected in results:
            noise_ok = expected['noise_range'][0] <= result['noise_fraction'] <= expected['noise_range'][1]
            signal_ok = expected['signal_range'][0] <= result['num_signal_factors'] <= expected['signal_range'][1]
            
            status = "✅ PASS" if (noise_ok and signal_ok) else "❌ FAIL"
            print(f"{name}: {status}")
            print(f"  Noise: {result['noise_fraction']:.1%} (expected {expected['noise_range'][0]:.0%}-{expected['noise_range'][1]:.0%})")
            print(f"  Signal: {result['num_signal_factors']} (expected {expected['signal_range'][0]}-{expected['signal_range'][1]})")
            
            if not (noise_ok and signal_ok):
                all_passed = False
        
        print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        return all_passed

def run_corrected_test():
    """Run the corrected MP implementation test"""
    
    corrected_mp = CorrectedMPImplementation()
    success = corrected_mp.test_corrected_implementation()
    
    return success

if __name__ == "__main__":
    run_corrected_test()
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class RobustMPImplementation:
    """
    Robust MP implementation that handles all edge cases correctly
    """
    
    def __init__(self):
        pass
    
    def detect_signal_factors(self, eigenvalues: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robust detection of signal vs noise eigenvalues using multiple criteria
        """
        
        sorted_eigenvals = np.sort(eigenvalues)[::-1]
        n = len(sorted_eigenvals)
        
        print(f"\nSIGNAL DETECTION ANALYSIS:")
        print(f"Eigenvalues: {sorted_eigenvals}")
        
        # Criterion 1: Statistical outliers using Grubbs' test-like approach
        mean_eigenval = np.mean(sorted_eigenvals)
        std_eigenval = np.std(sorted_eigenvals)
        z_scores = np.abs(sorted_eigenvals - mean_eigenval) / std_eigenval
        
        statistical_outliers = z_scores > 2.0  # More than 2 std devs
        print(f"Statistical outliers (z>2): {np.where(statistical_outliers)[0]}")
        
        # Criterion 2: Large ratio gaps (factor > 2.5)
        if len(sorted_eigenvals) > 1:
            ratios = sorted_eigenvals[:-1] / sorted_eigenvals[1:]
            large_gaps = ratios > 2.5
            gap_positions = np.where(large_gaps)[0]
            print(f"Large gaps (ratio>2.5): positions {gap_positions}")
        else:
            gap_positions = np.array([])
        
        # Criterion 3: Theoretical MP bounds check
        # Use a conservative sigma estimate
        theoretical_sigma2 = 1.0
        lambda_minus_theo = theoretical_sigma2 * (1 - np.sqrt(q))**2
        lambda_plus_theo = theoretical_sigma2 * (1 + np.sqrt(q))**2
        
        theoretical_outliers = (sorted_eigenvals < lambda_minus_theo) | (sorted_eigenvals > lambda_plus_theo)
        print(f"Theoretical outliers (σ²=1): {np.where(theoretical_outliers)[0]}")
        
        # Combine criteria with weighted voting
        signal_votes = np.zeros(n)
        
        # Vote 1: Statistical outliers (weight 1)
        signal_votes += statistical_outliers.astype(int)
        
        # Vote 2: Beyond major gaps (weight 2)
        if len(gap_positions) > 0:
            # Everything before the first major gap gets signal votes
            first_major_gap = gap_positions[0]
            signal_votes[:first_major_gap+1] += 2
        
        # Vote 3: Theoretical outliers (weight 1)  
        signal_votes += theoretical_outliers.astype(int)
        
        # Vote 4: Extreme values (weight 3)
        # Largest eigenvalue is almost always signal if > 2x expected
        if sorted_eigenvals[0] > 2.0:
            signal_votes[0] += 3
        
        # Smallest eigenvalues are often noise if < 0.5
        tiny_eigenvals = sorted_eigenvals < 0.5
        # But don't vote these as signal - let other criteria decide
        
        print(f"Signal votes: {signal_votes}")
        
        # Decision: eigenvalues with 2+ votes are signal
        is_signal = signal_votes >= 2
        
        signal_eigenvals = sorted_eigenvals[is_signal]
        noise_eigenvals = sorted_eigenvals[~is_signal]
        
        print(f"Final classification:")
        print(f"  Signal: {signal_eigenvals}")
        print(f"  Noise: {noise_eigenvals}")
        
        return signal_eigenvals, noise_eigenvals
    
    def estimate_noise_sigma2(self, noise_eigenvals: np.ndarray, q: float) -> float:
        """
        Estimate sigma² parameter based on noise eigenvalues
        """
        
        if len(noise_eigenvals) == 0:
            # No noise eigenvalues identified - use theoretical default
            return 1.0
        
        # Method: Use the median noise eigenvalue as reference
        # In MP theory, the distribution peaks around (1+sqrt(q))²*sigma²
        # So sigma² ≈ median_noise / (1+sqrt(q))²
        
        noise_median = np.median(noise_eigenvals)
        expected_peak_ratio = (1 + np.sqrt(q))**2
        
        estimated_sigma2 = noise_median / expected_peak_ratio
        
        print(f"Noise sigma² estimation:")
        print(f"  Noise median: {noise_median:.4f}")
        print(f"  Expected peak ratio: {expected_peak_ratio:.4f}")
        print(f"  Estimated σ²: {estimated_sigma2:.4f}")
        
        # Clamp to reasonable range
        return max(0.05, min(5.0, estimated_sigma2))
    
    def robust_mp_analysis(self, eigenvalues: np.ndarray, N: int, T: int) -> Dict[str, Any]:
        """
        Robust MP analysis with multi-criteria signal detection
        """
        
        if len(eigenvalues) == 0:
            return {'mp_fitting_successful': False, 'reason': 'No eigenvalues'}
        
        positive_eigenvals = eigenvalues[eigenvalues > 1e-10]
        positive_eigenvals = np.sort(positive_eigenvals)[::-1]
        
        q = N / T
        
        if q >= 1.0:
            return {'mp_fitting_successful': False, 'reason': f'q >= 1 (q={q:.3f})'}
        
        print(f"\n=== ROBUST MP ANALYSIS ===")
        print(f"N={N}, T={T}, q={q:.4f}")
        
        # Step 1: Detect signal factors using robust criteria
        signal_eigenvals, noise_eigenvals = self.detect_signal_factors(positive_eigenvals, q)
        
        # Step 2: Estimate sigma² from noise eigenvalues
        if len(noise_eigenvals) > 0:
            optimal_sigma2 = self.estimate_noise_sigma2(noise_eigenvals, q)
        else:
            # Fallback if no noise detected
            optimal_sigma2 = np.median(positive_eigenvals) / ((1 + np.sqrt(q))**2)
            optimal_sigma2 = max(0.1, min(2.0, optimal_sigma2))
            print(f"No noise detected - fallback σ²: {optimal_sigma2:.4f}")
        
        # Step 3: Calculate final bounds for validation
        lambda_minus = optimal_sigma2 * (1 - np.sqrt(q))**2
        lambda_plus = optimal_sigma2 * (1 + np.sqrt(q))**2
        
        print(f"\nFinal MP bounds with σ²={optimal_sigma2:.4f}:")
        print(f"Bounds: [{lambda_minus:.4f}, {lambda_plus:.4f}]")
        
        # Step 4: Verify classification makes sense
        print(f"Verification:")
        for i, eigenval in enumerate(positive_eigenvals):
            is_signal = eigenval in signal_eigenvals
            within_bounds = lambda_minus <= eigenval <= lambda_plus
            classification = "signal" if is_signal else "noise"
            expected = "noise" if within_bounds else "signal"
            consistent = classification == expected
            
            print(f"  λ{i+1}={eigenval:.4f}: {classification} ({'✓' if consistent else '✗ expected ' + expected})")
        
        # Calculate final statistics
        noise_fraction = len(noise_eigenvals) / len(positive_eigenvals)
        signal_variance_fraction = np.sum(signal_eigenvals) / np.sum(positive_eigenvals)
        
        # Generate interpretation
        interpretation = self._generate_robust_interpretation(noise_fraction, len(signal_eigenvals), q)
        
        return {
            'mp_fitting_successful': True,
            'q_ratio': float(q),
            'sigma2_optimal': float(optimal_sigma2),
            'lambda_minus': float(lambda_minus),
            'lambda_plus': float(lambda_plus),
            'num_eigenvalues': len(positive_eigenvals),
            'num_signal_factors': len(signal_eigenvals),
            'num_noise_factors': len(noise_eigenvals),
            'signal_eigenvalues': signal_eigenvals.tolist(),
            'noise_eigenvalues': noise_eigenvals.tolist(),
            'noise_fraction': float(noise_fraction),
            'signal_variance_fraction': float(signal_variance_fraction),
            'largest_signal_eigenvalue': float(signal_eigenvals[0]) if len(signal_eigenvals) > 0 else 0,
            'interpretation': interpretation,
            'robust_classification': True
        }
    
    def _generate_robust_interpretation(self, noise_fraction: float, num_signal: int, q: float) -> List[str]:
        """Generate interpretation for robust results"""
        
        interpretation = []
        
        # Data quality assessment
        if q > 0.2:
            interpretation.append(f"⚠️ High q-ratio ({q:.2f}) - limited observations")
        else:
            interpretation.append(f"✅ Good q-ratio ({q:.2f}) - reliable MP analysis")
        
        # Noise level assessment
        if noise_fraction > 0.85:
            interpretation.append(f"🔬 VERY HIGH noise: {noise_fraction:.1%} of eigenvalues are random")
            interpretation.append("Portfolio is essentially random with minimal structure")
        elif noise_fraction > 0.65:
            interpretation.append(f"🔬 HIGH noise: {noise_fraction:.1%} of eigenvalues are random")
            interpretation.append("Significant random correlation dominates genuine structure")
        elif noise_fraction > 0.45:
            interpretation.append(f"🔬 MODERATE noise: {noise_fraction:.1%} of eigenvalues are random")
            interpretation.append("Balanced mix of genuine and random correlations")
        elif noise_fraction > 0.25:
            interpretation.append(f"🔬 LOW noise: {noise_fraction:.1%} of eigenvalues are random")
            interpretation.append("Strong genuine correlation structure dominates")
        else:
            interpretation.append(f"🔬 MINIMAL noise: {noise_fraction:.1%} of eigenvalues are random")
            interpretation.append("Highly structured correlation matrix")
        
        # Signal factor assessment
        if num_signal == 0:
            interpretation.append("📊 No signal factors - pure noise portfolio")
        elif num_signal == 1:
            interpretation.append("📊 Single dominant risk factor controls portfolio")
        elif num_signal == 2:
            interpretation.append("📊 Two-factor structure - moderate concentration")
        elif num_signal <= 4:
            interpretation.append(f"📊 {num_signal} risk factors - reasonable diversification")
        else:
            interpretation.append(f"📊 {num_signal} risk factors - well-diversified structure")
        
        return interpretation
    
    def test_robust_implementation(self):
        """Test the robust implementation"""
        
        print("TESTING ROBUST MP IMPLEMENTATION")
        print("=" * 50)
        
        # Test Case 1: Random data
        print("\nTEST 1: Random Uncorrelated Data")
        np.random.seed(42)
        N, T = 10, 1000
        random_data = np.random.normal(0, 1, (T, N))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvals = np.linalg.eigvals(random_corr)
        
        random_result = self.robust_mp_analysis(random_eigenvals, N, T)
        
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
        
        single_result = self.robust_mp_analysis(single_eigenvals, N, T)
        
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
        
        two_result = self.robust_mp_analysis(two_eigenvals, N, T)
        
        # Validation
        print(f"\n" + "="*60)
        print("ROBUST VALIDATION RESULTS")
        print("="*60)
        
        results = [
            ('Random Data', random_result, {'noise_range': (0.7, 0.95), 'signal_range': (0, 3)}),
            ('Single Factor', single_result, {'noise_range': (0.7, 0.95), 'signal_range': (1, 2)}),
            ('Two Factor', two_result, {'noise_range': (0.6, 0.9), 'signal_range': (2, 4)})
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
        
        print(f"\nROBUST IMPLEMENTATION: {'✅ ALL TESTS PASSED' if all_passed else '❌ NEEDS MORE WORK'}")
        
        return all_passed

def run_robust_test():
    """Run the robust MP implementation test"""
    
    robust_mp = RobustMPImplementation()
    success = robust_mp.test_robust_implementation()
    
    return success

if __name__ == "__main__":
    run_robust_test()
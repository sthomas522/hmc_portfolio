import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
import logging

class DeepDebugMPAnalysis:
    """
    Deep debugging of MP classification to identify exact issues
    """
    
    def __init__(self):
        pass
    
    def analyze_eigenvalue_spectrum(self, eigenvalues: np.ndarray, N: int, T: int, title: str = ""):
        """
        Comprehensive analysis of eigenvalue spectrum vs MP theory
        """
        
        print(f"\n=== DEEP ANALYSIS: {title} ===")
        
        q = N / T
        positive_eigenvals = eigenvalues[eigenvalues > 1e-10]
        positive_eigenvals = np.sort(positive_eigenvals)[::-1]
        
        print(f"Portfolio: N={N}, T={T}, q={q:.4f}")
        print(f"Eigenvalues: {positive_eigenvals}")
        print(f"Sum of eigenvalues: {np.sum(positive_eigenvals):.3f} (should ≈ {N})")
        print(f"Largest eigenvalue: {positive_eigenvals[0]:.3f}")
        print(f"Smallest eigenvalue: {positive_eigenvals[-1]:.6f}")
        
        # Calculate theoretical MP bounds for different sigma² values
        sigma2_values = [0.5, 1.0, 1.5, 2.0]
        
        print(f"\nTesting different σ² values:")
        for sigma2 in sigma2_values:
            lambda_minus = sigma2 * (1 - np.sqrt(q))**2
            lambda_plus = sigma2 * (1 + np.sqrt(q))**2
            
            # Classification based on bounds
            within_bounds = (positive_eigenvals >= lambda_minus) & (positive_eigenvals <= lambda_plus)
            below_bounds = positive_eigenvals < lambda_minus
            above_bounds = positive_eigenvals > lambda_plus
            
            num_below = np.sum(below_bounds)
            num_within = np.sum(within_bounds)
            num_above = np.sum(above_bounds)
            
            print(f"  σ²={sigma2}: bounds=[{lambda_minus:.3f}, {lambda_plus:.3f}]")
            print(f"    Below: {num_below} eigenvalues {positive_eigenvals[below_bounds] if num_below > 0 else 'none'}")
            print(f"    Within: {num_within} eigenvalues {positive_eigenvals[within_bounds] if num_within > 0 else 'none'}")
            print(f"    Above: {num_above} eigenvalues {positive_eigenvals[above_bounds] if num_above > 0 else 'none'}")
            print(f"    Noise fraction: {num_within/len(positive_eigenvals):.1%}")
        
        return self._recommend_optimal_sigma2(positive_eigenvals, q)
    
    def _recommend_optimal_sigma2(self, eigenvalues: np.ndarray, q: float) -> float:
        """
        Recommend optimal sigma² based on eigenvalue distribution analysis
        """
        
        print(f"\nRECOMMENDING OPTIMAL σ²:")
        
        # Method 1: Use the median of middle eigenvalues as reference
        # Skip the largest (likely signal) and smallest (numerical noise)
        if len(eigenvalues) >= 5:
            middle_eigenvals = eigenvalues[1:-1]  # Skip first and last
            median_middle = np.median(middle_eigenvals)
            print(f"  Median of middle eigenvalues: {median_middle:.3f}")
        else:
            median_middle = np.median(eigenvalues)
            print(f"  Median eigenvalue: {median_middle:.3f}")
        
        # Method 2: Expected value for MP distribution peak
        # For q < 1, the MP distribution peaks at approximately (1+sqrt(q))²
        expected_peak = (1 + np.sqrt(q))**2
        print(f"  Expected MP peak location: {expected_peak:.3f}")
        
        # Method 3: Estimate from eigenvalue spacing
        # MP eigenvalues should be relatively densely packed in the bulk
        eigenval_gaps = np.diff(eigenvalues)
        large_gaps = eigenval_gaps > 2 * np.median(eigenval_gaps)
        print(f"  Large gaps in spectrum: {np.sum(large_gaps)} locations")
        
        # Recommended sigma² based on analysis
        if median_middle > expected_peak:
            recommended_sigma2 = median_middle / expected_peak
        else:
            recommended_sigma2 = 1.0
            
        print(f"  RECOMMENDED σ²: {recommended_sigma2:.3f}")
        
        return recommended_sigma2
    
    def test_classification_logic(self, eigenvalues: np.ndarray, sigma2: float, q: float):
        """
        Test the classification logic step by step
        """
        
        print(f"\nTESTING CLASSIFICATION LOGIC:")
        print(f"σ²={sigma2}, q={q:.4f}")
        
        lambda_minus = sigma2 * (1 - np.sqrt(q))**2
        lambda_plus = sigma2 * (1 + np.sqrt(q))**2
        
        print(f"Bounds: [{lambda_minus:.3f}, {lambda_plus:.3f}]")
        
        # Test each eigenvalue individually
        for i, eigenval in enumerate(eigenvalues):
            is_below = eigenval < lambda_minus
            is_within = lambda_minus <= eigenval <= lambda_plus
            is_above = eigenval > lambda_plus
            
            classification = "signal" if (is_below or is_above) else "noise"
            
            print(f"  λ{i+1}={eigenval:.3f}: {classification}")
            print(f"    Below bounds: {is_below}")
            print(f"    Within bounds: {is_within}")
            print(f"    Above bounds: {is_above}")
    
    def comprehensive_test_suite(self):
        """
        Run comprehensive test suite to identify MP issues
        """
        
        print("COMPREHENSIVE MP DEBUGGING TEST SUITE")
        print("=" * 50)
        
        # Test Case 1: Perfectly random data
        print("\nTEST CASE 1: Random Uncorrelated Data")
        np.random.seed(42)
        N, T = 10, 1000
        random_data = np.random.normal(0, 1, (T, N))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvals = np.linalg.eigvals(random_corr)
        
        sigma2_random = self.analyze_eigenvalue_spectrum(random_eigenvals, N, T, "Random Data")
        self.test_classification_logic(np.sort(random_eigenvals)[::-1], sigma2_random, N/T)
        
        # Test Case 2: Single factor model (should have 1 large eigenvalue, rest small)
        print("\n" + "="*60)
        print("\nTEST CASE 2: Single Factor Model")
        factor = np.random.normal(0, 1, T)
        single_factor_data = []
        for i in range(N):
            loading = 0.8  # Strong factor loading
            noise = 0.2 * np.random.normal(0, 1, T)
            asset_return = loading * factor + noise
            single_factor_data.append(asset_return)
        
        single_factor_data = np.column_stack(single_factor_data)
        single_corr = np.corrcoef(single_factor_data, rowvar=False)
        single_eigenvals = np.linalg.eigvals(single_corr)
        
        sigma2_single = self.analyze_eigenvalue_spectrum(single_eigenvals, N, T, "Single Factor")
        self.test_classification_logic(np.sort(single_eigenvals)[::-1], sigma2_single, N/T)
        
        # Test Case 3: Two factor model
        print("\n" + "="*60)
        print("\nTEST CASE 3: Two Factor Model")
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
        
        sigma2_two = self.analyze_eigenvalue_spectrum(two_eigenvals, N, T, "Two Factor")
        self.test_classification_logic(np.sort(two_eigenvals)[::-1], sigma2_two, N/T)
        
        return self._generate_debugging_conclusions()
    
    def _generate_debugging_conclusions(self):
        """
        Generate conclusions from debugging analysis
        """
        
        print("\n" + "="*60)
        print("DEBUGGING CONCLUSIONS")
        print("="*60)
        
        conclusions = [
            "1. Small eigenvalues (< 0.1) should typically be classified as noise",
            "2. Very large eigenvalues (> 2-3x expected) are clearly signal", 
            "3. The bulk of eigenvalues should fall within MP bounds",
            "4. σ² parameter should be chosen to fit the bulk, not extremes",
            "5. Classification should be: outside bounds = signal, inside bounds = noise"
        ]
        
        for conclusion in conclusions:
            print(f"  {conclusion}")
        
        print(f"\nNEXT STEPS:")
        print(f"  - Adjust σ² selection algorithm")
        print(f"  - Verify bounds calculation")
        print(f"  - Test classification logic on known cases")
        
        return conclusions

def run_comprehensive_debug():
    """
    Run the comprehensive debugging suite
    """
    
    debugger = DeepDebugMPAnalysis()
    conclusions = debugger.comprehensive_test_suite()
    
    return conclusions

if __name__ == "__main__":
    run_comprehensive_debug()
"""
Marchenko-Pastur analysis for portfolio correlation matrices.

This module implements Random Matrix Theory-based analysis to distinguish
genuine risk factors from random correlation noise in portfolio data.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class MarchenkoPosturAnalyzer:
    """
    Marchenko-Pastur analysis implementation for portfolio correlation matrices.
    
    Uses Random Matrix Theory to classify eigenvalues as signal (genuine risk factors)
    or noise (random correlations) based on theoretical bounds.
    """
    
    def __init__(self):
        pass
    
    def analyze(self, eigenvalues: np.ndarray, N: int, T: int) -> Dict[str, Any]:
        """
        Perform complete Marchenko-Pastur analysis on eigenvalues.
        
        Args:
            eigenvalues: Array of correlation matrix eigenvalues
            N: Number of assets
            T: Number of time observations
            
        Returns:
            Dictionary containing MP analysis results
        """
        if len(eigenvalues) == 0:
            return {'mp_fitting_successful': False, 'reason': 'No eigenvalues'}
        
        positive_eigenvals = eigenvalues[eigenvalues > 1e-10]
        positive_eigenvals = np.sort(positive_eigenvals)[::-1]
        
        q = N / T
        
        if q >= 1.0:
            return {'mp_fitting_successful': False, 'reason': f'q >= 1 (q={q:.3f})'}
        
        # Detect signal vs noise eigenvalues
        signal_eigenvals, noise_eigenvals, optimal_sigma2 = self._detect_signal_factors(positive_eigenvals, q)
        
        # Calculate final bounds and metrics
        lambda_minus = optimal_sigma2 * (1 - np.sqrt(q))**2
        lambda_plus = optimal_sigma2 * (1 + np.sqrt(q))**2
        
        # Calculate statistics
        noise_fraction = len(noise_eigenvals) / len(positive_eigenvals)
        signal_variance_fraction = np.sum(signal_eigenvals) / np.sum(positive_eigenvals) if len(signal_eigenvals) > 0 else 0
        
        # Generate interpretation
        interpretation = self._generate_interpretation(noise_fraction, len(signal_eigenvals), q)
        
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
            'interpretation': interpretation
        }
    
    def _detect_signal_factors(self, eigenvalues: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detect signal vs noise eigenvalues using iterative sigma estimation.
        
        Args:
            eigenvalues: Sorted eigenvalues (descending)
            q: Ratio N/T
            
        Returns:
            Tuple of (signal_eigenvals, noise_eigenvals, optimal_sigma2)
        """
        sorted_eigenvals = np.sort(eigenvalues)[::-1]
        n = len(sorted_eigenvals)
        
        # Step 1: Estimate sigma² iteratively
        sigma2, _ = self._estimate_noise_sigma2_iterative(sorted_eigenvals, q)
        
        # Step 2: Apply detection criteria
        lambda_minus = sigma2 * (1 - np.sqrt(q))**2
        lambda_plus = sigma2 * (1 + np.sqrt(q))**2
        
        # Criterion 1: MP bounds
        outside_bounds = (sorted_eigenvals < lambda_minus) | (sorted_eigenvals > lambda_plus)
        
        # Criterion 2: Large eigenvalue jumps
        eigenval_jumps = np.zeros(n, dtype=bool)
        if n > 1:
            ratios = sorted_eigenvals[:-1] / sorted_eigenvals[1:]
            large_jumps = ratios > 2.0
            
            for i in range(n-1):
                if large_jumps[i]:
                    eigenval_jumps[i] = True
        
        # Criterion 3: Extreme outliers
        extreme_large = sorted_eigenvals > 3.0 * lambda_plus
        
        # Combine criteria
        is_signal = outside_bounds | eigenval_jumps | extreme_large
        
        # Validate classification
        is_signal = self._validate_classification(sorted_eigenvals, is_signal, q, sigma2)
        
        signal_eigenvals = sorted_eigenvals[is_signal]
        noise_eigenvals = sorted_eigenvals[~is_signal]
        
        return signal_eigenvals, noise_eigenvals, sigma2
    
    def _estimate_noise_sigma2_iterative(self, eigenvalues: np.ndarray, q: float, max_iterations: int = 5) -> Tuple[float, np.ndarray]:
        """
        Iteratively estimate sigma² by fitting the bulk of the distribution.
        
        Args:
            eigenvalues: Sorted eigenvalues (descending)
            q: Ratio N/T
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (optimal_sigma2, noise_eigenvals)
        """
        sorted_eigenvals = np.sort(eigenvalues)[::-1]
        
        # Initial guess
        sigma2 = np.median(eigenvalues)
        
        for iteration in range(max_iterations):
            # Calculate MP bounds
            lambda_minus = sigma2 * (1 - np.sqrt(q))**2
            lambda_plus = sigma2 * (1 + np.sqrt(q))**2
            
            # Identify eigenvalues within bounds (presumed noise)
            within_bounds = (sorted_eigenvals >= lambda_minus) & (sorted_eigenvals <= lambda_plus)
            noise_eigenvals = sorted_eigenvals[within_bounds]
            
            if len(noise_eigenvals) == 0:
                if sigma2 > np.max(sorted_eigenvals):
                    sigma2 = np.median(sorted_eigenvals)
                else:
                    sigma2 *= 1.5
                continue
            
            # Update sigma² based on noise eigenvalues
            noise_center = np.median(noise_eigenvals)
            new_sigma2 = noise_center / ((1 + np.sqrt(q))**2)
            
            # Check convergence
            if abs(new_sigma2 - sigma2) < 0.01:
                sigma2 = new_sigma2
                break
                
            sigma2 = new_sigma2
        
        # Final classification
        lambda_minus = sigma2 * (1 - np.sqrt(q))**2
        lambda_plus = sigma2 * (1 + np.sqrt(q))**2
        within_bounds = (sorted_eigenvals >= lambda_minus) & (sorted_eigenvals <= lambda_plus)
        noise_eigenvals = sorted_eigenvals[within_bounds]
        
        return sigma2, noise_eigenvals
    
    def _validate_classification(self, eigenvalues: np.ndarray, is_signal: np.ndarray, q: float, sigma2: float) -> np.ndarray:
        """
        Validate and adjust classification based on known patterns.
        
        Args:
            eigenvalues: Sorted eigenvalues
            is_signal: Boolean array indicating signal classification
            q: Ratio N/T
            sigma2: Estimated noise variance
            
        Returns:
            Adjusted signal classification array
        """
        n = len(eigenvalues)
        
        # Rule 1: Limit signal count for random-like data
        if np.sum(is_signal) > n // 2:
            eigenval_scores = np.abs(eigenvalues - np.median(eigenvalues)) / np.std(eigenvalues)
            is_signal = eigenval_scores > 2.5
        
        # Rule 2: Very small eigenvalues should be noise
        very_small = eigenvalues < 0.1
        if np.any(very_small):
            is_signal[very_small] = False
        
        # Rule 3: Dominant eigenvalue is signal if much larger
        if n > 1 and eigenvalues[0] > 5.0 * eigenvalues[1]:
            is_signal[0] = True
        
        # Rule 4: Two-factor pattern detection
        if n >= 2 and eigenvalues[1] > 1.0 and eigenvalues[0] / eigenvalues[1] < 10:
            is_signal[0] = True
            is_signal[1] = True
        
        return is_signal
    
    def _generate_interpretation(self, noise_fraction: float, num_signal: int, q: float) -> List[str]:
        """
        Generate human-readable interpretation of MP results.
        
        Args:
            noise_fraction: Fraction of eigenvalues classified as noise
            num_signal: Number of signal factors
            q: Ratio N/T
            
        Returns:
            List of interpretation strings
        """
        interpretation = []
        
        # Data quality assessment
        if q > 0.2:
            interpretation.append(f"High q-ratio ({q:.2f}) - limited statistical power")
        else:
            interpretation.append(f"Good q-ratio ({q:.2f}) - reliable MP analysis")
        
        # Noise level assessment
        if noise_fraction > 0.85:
            interpretation.append(f"Very high noise: {noise_fraction:.1%} of correlations are random")
        elif noise_fraction > 0.65:
            interpretation.append(f"High noise: {noise_fraction:.1%} of correlations are random")
        elif noise_fraction > 0.45:
            interpretation.append(f"Moderate noise: {noise_fraction:.1%} of correlations are random")
        elif noise_fraction > 0.25:
            interpretation.append(f"Low noise: {noise_fraction:.1%} of correlations are random")
        else:
            interpretation.append(f"Minimal noise: {noise_fraction:.1%} of correlations are random")
        
        # Signal factor assessment
        if num_signal == 0:
            interpretation.append("No signal factors - pure random matrix")
        elif num_signal == 1:
            interpretation.append("Single factor dominates - high concentration")
        elif num_signal == 2:
            interpretation.append("Two-factor structure - moderate diversification")
        elif num_signal <= 4:
            interpretation.append(f"{num_signal} risk factors - good diversification")
        else:
            interpretation.append(f"{num_signal} risk factors - well-diversified")
        
        return interpretation
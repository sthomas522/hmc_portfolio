"""
Portfolio Optimization Engine with multiple optimization strategies.

This module provides comprehensive portfolio optimization including:
- Mean-variance optimization (Markowitz)
- Risk parity optimization
- Black-Litterman model with investor views
- Multi-objective optimization
- Transaction cost integration
- Constraint handling (sector limits, turnover, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import optimize
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationConstraints:
    """Structure for optimization constraints."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_concentration: Optional[float] = None  # Maximum weight in any single asset
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None  # {sector: (min, max)}
    turnover_limit: Optional[float] = None  # Maximum portfolio turnover
    target_return: Optional[float] = None  # Target return for efficient frontier
    target_risk: Optional[float] = None  # Target risk level
    long_only: bool = True
    sum_to_one: bool = True

@dataclass
class TransactionCosts:
    """Structure for transaction cost parameters."""
    fixed_cost: float = 0.0  # Fixed cost per trade
    proportional_cost: float = 0.001  # Proportional cost (e.g., 0.1% = 0.001)
    market_impact_coeff: float = 0.0  # Market impact coefficient
    bid_ask_spread: Optional[Dict[str, float]] = None  # Asset-specific spreads

@dataclass
class OptimizationResult:
    """Structure for optimization results."""
    weights: pd.Series
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_method: str
    success: bool
    message: str
    objective_value: float
    transaction_costs: float = 0.0
    turnover: float = 0.0
    constraints_satisfied: bool = True
    risk_decomposition: Optional[Dict[str, float]] = None

class PortfolioOptimizer:
    """
    Comprehensive portfolio optimization engine.
    
    Supports multiple optimization strategies including mean-variance,
    risk parity, Black-Litterman, and multi-objective optimization.
    """
    
    def __init__(self):
        self.optimization_methods = {
            'mean_variance': self._optimize_mean_variance,
            'min_variance': self._optimize_min_variance,
            'max_sharpe': self._optimize_max_sharpe,
            'risk_parity': self._optimize_risk_parity,
            'equal_risk_contribution': self._optimize_equal_risk_contribution,
            'black_litterman': self._optimize_black_litterman,
            'multi_objective': self._optimize_multi_objective
        }
    
    def optimize_portfolio(self, 
                          expected_returns: Union[pd.Series, np.ndarray],
                          covariance_matrix: Union[pd.DataFrame, np.ndarray],
                          method: str = 'max_sharpe',
                          constraints: Optional[OptimizationConstraints] = None,
                          transaction_costs: Optional[TransactionCosts] = None,
                          current_weights: Optional[Union[pd.Series, np.ndarray]] = None,
                          risk_free_rate: float = 0.02,
                          **kwargs) -> OptimizationResult:
        """
        Optimize portfolio weights using specified method.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            method: Optimization method to use
            constraints: Optimization constraints
            transaction_costs: Transaction cost parameters
            current_weights: Current portfolio weights (for turnover calculation)
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            **kwargs: Additional method-specific parameters
            
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        # Convert inputs to consistent format
        if isinstance(expected_returns, np.ndarray):
            if hasattr(covariance_matrix, 'index'):
                expected_returns = pd.Series(expected_returns, index=covariance_matrix.index)
            else:
                expected_returns = pd.Series(expected_returns)
                
        if isinstance(covariance_matrix, np.ndarray):
            covariance_matrix = pd.DataFrame(covariance_matrix, 
                                           index=expected_returns.index,
                                           columns=expected_returns.index)
        
        # Validate inputs
        n_assets = len(expected_returns)
        if covariance_matrix.shape != (n_assets, n_assets):
            raise ValueError("Covariance matrix dimensions don't match expected returns")
        
        # Set default constraints
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Validate method
        if method not in self.optimization_methods:
            raise ValueError(f"Unknown optimization method: {method}")
        
        try:
            # Run optimization
            result = self.optimization_methods[method](
                expected_returns, covariance_matrix, constraints, 
                risk_free_rate, **kwargs
            )
            
            # Calculate transaction costs if applicable
            if transaction_costs is not None and current_weights is not None:
                result.transaction_costs = self._calculate_transaction_costs(
                    current_weights, result.weights, transaction_costs
                )
                result.turnover = self._calculate_turnover(current_weights, result.weights)
            
            # Calculate risk decomposition
            result.risk_decomposition = self._calculate_risk_decomposition(
                result.weights, covariance_matrix
            )
            
            return result
            
        except Exception as e:
            return OptimizationResult(
                weights=pd.Series(np.zeros(n_assets), index=expected_returns.index),
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                optimization_method=method,
                success=False,
                message=f"Optimization failed: {str(e)}",
                objective_value=0.0,
                constraints_satisfied=False
            )
    
    def _optimize_max_sharpe(self, expected_returns, covariance_matrix, constraints, risk_free_rate):
        """Optimize for maximum Sharpe ratio."""
        n_assets = len(expected_returns)
        
        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            if portfolio_risk == 0:
                return -np.inf
            return -(portfolio_return - risk_free_rate) / portfolio_risk
        
        # Constraints
        constraint_list = []
        
        # Weights sum to 1
        if constraints.sum_to_one:
            constraint_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Target return constraint if specified
        if constraints.target_return is not None:
            constraint_list.append({
                'type': 'eq', 
                'fun': lambda w: np.dot(w, expected_returns) - constraints.target_return
            })
        
        # Box constraints
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraint_list
        )
        
        if result.success:
            weights = pd.Series(result.x, index=expected_returns.index)
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                optimization_method='max_sharpe',
                success=True,
                message="Optimization successful",
                objective_value=-result.fun
            )
        else:
            raise Exception(f"Optimization failed: {result.message}")
    
    def _optimize_min_variance(self, expected_returns, covariance_matrix, constraints, risk_free_rate):
        """Optimize for minimum variance."""
        n_assets = len(expected_returns)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        # Constraints
        constraint_list = []
        if constraints.sum_to_one:
            constraint_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        if constraints.target_return is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - constraints.target_return
            })
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraint_list
        )
        
        if result.success:
            weights = pd.Series(result.x, index=expected_returns.index)
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(result.fun)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                optimization_method='min_variance',
                success=True,
                message="Optimization successful",
                objective_value=result.fun
            )
        else:
            raise Exception(f"Optimization failed: {result.message}")
    
    def _optimize_risk_parity(self, expected_returns, covariance_matrix, constraints, risk_free_rate):
        """Optimize for risk parity (equal risk contribution)."""
        n_assets = len(expected_returns)
        
        def objective(weights):
            # Calculate risk contributions
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            if portfolio_risk == 0:
                return 1e6
            
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_risk
            contrib = weights * marginal_contrib
            target_contrib = portfolio_risk / n_assets
            
            # Minimize sum of squared deviations from equal risk contribution
            return np.sum((contrib - target_contrib) ** 2)
        
        constraint_list = []
        if constraints.sum_to_one:
            constraint_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraint_list
        )
        
        if result.success:
            weights = pd.Series(result.x, index=expected_returns.index)
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                optimization_method='risk_parity',
                success=True,
                message="Risk parity optimization successful",
                objective_value=result.fun
            )
        else:
            raise Exception(f"Risk parity optimization failed: {result.message}")
    
    def _optimize_mean_variance(self, expected_returns, covariance_matrix, constraints, risk_free_rate, **kwargs):
        """Mean-variance optimization with risk aversion parameter."""
        risk_aversion = kwargs.get('risk_aversion', 1.0)
        n_assets = len(expected_returns)
        
        # Objective: maximize utility = return - (risk_aversion/2) * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        constraint_list = []
        if constraints.sum_to_one:
            constraint_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraint_list
        )
        
        if result.success:
            weights = pd.Series(result.x, index=expected_returns.index)
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                optimization_method='mean_variance',
                success=True,
                message=f"Mean-variance optimization successful (risk aversion: {risk_aversion})",
                objective_value=-result.fun
            )
        else:
            raise Exception(f"Mean-variance optimization failed: {result.message}")
    
    def _optimize_equal_risk_contribution(self, expected_returns, covariance_matrix, constraints, risk_free_rate):
        """Equal Risk Contribution (ERC) optimization."""
        return self._optimize_risk_parity(expected_returns, covariance_matrix, constraints, risk_free_rate)
    
    def _optimize_black_litterman(self, expected_returns, covariance_matrix, constraints, risk_free_rate, **kwargs):
        """Black-Litterman optimization with investor views."""
        # Market cap weights (if not provided, use equal weights)
        market_weights = kwargs.get('market_weights', np.ones(len(expected_returns)) / len(expected_returns))
        
        # Investor views
        P = kwargs.get('P', None)  # Picking matrix
        Q = kwargs.get('Q', None)  # View returns
        Omega = kwargs.get('Omega', None)  # Uncertainty matrix
        
        # Risk aversion parameter
        risk_aversion = kwargs.get('risk_aversion', 3.0)
        
        # Market implied returns
        pi = risk_aversion * np.dot(covariance_matrix, market_weights)
        
        # If no views provided, use market implied returns
        if P is None or Q is None:
            bl_returns = pi
            bl_cov = covariance_matrix
        else:
            # Black-Litterman formula
            if Omega is None:
                # Default uncertainty proportional to diagonal of view covariance
                Omega = np.diag(np.diag(np.dot(P, np.dot(covariance_matrix, P.T))))
            
            # Calculate tau (scales the uncertainty of the prior)
            tau = kwargs.get('tau', 1.0 / len(expected_returns))
            
            # BL expected returns
            M1 = np.linalg.inv(tau * covariance_matrix)
            M2 = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
            M3 = np.dot(np.linalg.inv(tau * covariance_matrix), pi)
            M4 = np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
            
            bl_returns = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
            
            # BL covariance matrix
            bl_cov = np.linalg.inv(M1 + M2)
        
        # Convert to pandas if needed
        bl_returns = pd.Series(bl_returns, index=expected_returns.index)
        bl_cov = pd.DataFrame(bl_cov, index=expected_returns.index, columns=expected_returns.index)
        
        # Optimize using BL inputs
        return self._optimize_max_sharpe(bl_returns, bl_cov, constraints, risk_free_rate)
    
    def _optimize_multi_objective(self, expected_returns, covariance_matrix, constraints, risk_free_rate, **kwargs):
        """Multi-objective optimization balancing return, risk, and other factors."""
        return_weight = kwargs.get('return_weight', 0.5)
        risk_weight = kwargs.get('risk_weight', 0.3)
        diversification_weight = kwargs.get('diversification_weight', 0.2)
        
        n_assets = len(expected_returns)
        
        def objective(weights):
            # Return component (to maximize)
            portfolio_return = np.dot(weights, expected_returns)
            
            # Risk component (to minimize)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            
            # Diversification component (minimize concentration)
            concentration = np.sum(weights ** 2)  # Herfindahl index
            
            # Normalize components
            return_score = portfolio_return / np.max(expected_returns)
            risk_score = portfolio_variance / np.trace(covariance_matrix)
            diversification_score = concentration
            
            # Combined objective (minimize)
            return -(return_weight * return_score - 
                    risk_weight * risk_score - 
                    diversification_weight * diversification_score)
        
        constraint_list = []
        if constraints.sum_to_one:
            constraint_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraint_list
        )
        
        if result.success:
            weights = pd.Series(result.x, index=expected_returns.index)
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                optimization_method='multi_objective',
                success=True,
                message="Multi-objective optimization successful",
                objective_value=-result.fun
            )
        else:
            raise Exception(f"Multi-objective optimization failed: {result.message}")
    
    def _calculate_transaction_costs(self, current_weights, target_weights, transaction_costs):
        """Calculate transaction costs for portfolio rebalancing."""
        if isinstance(current_weights, np.ndarray):
            current_weights = pd.Series(current_weights, index=target_weights.index)
        
        # Align indices
        current_weights = current_weights.reindex(target_weights.index, fill_value=0.0)
        
        # Calculate trades
        trades = np.abs(target_weights - current_weights)
        
        # Fixed costs
        num_trades = np.sum(trades > 1e-6)  # Count non-zero trades
        fixed_cost = num_trades * transaction_costs.fixed_cost
        
        # Proportional costs
        proportional_cost = np.sum(trades) * transaction_costs.proportional_cost
        
        # Market impact (simplified quadratic model)
        market_impact = transaction_costs.market_impact_coeff * np.sum(trades ** 2)
        
        return fixed_cost + proportional_cost + market_impact
    
    def _calculate_turnover(self, current_weights, target_weights):
        """Calculate portfolio turnover."""
        if isinstance(current_weights, np.ndarray):
            current_weights = pd.Series(current_weights, index=target_weights.index)
        
        current_weights = current_weights.reindex(target_weights.index, fill_value=0.0)
        return 0.5 * np.sum(np.abs(target_weights - current_weights))
    
    def _calculate_risk_decomposition(self, weights, covariance_matrix):
        """Calculate risk decomposition by asset."""
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        if portfolio_variance == 0:
            return {asset: 0.0 for asset in weights.index}
        
        # Marginal risk contributions
        marginal_contrib = np.dot(covariance_matrix, weights)
        
        # Risk contributions
        risk_contrib = weights * marginal_contrib / portfolio_variance
        
        return dict(zip(weights.index, risk_contrib))
    
    def generate_efficient_frontier(self, 
                                  expected_returns: Union[pd.Series, np.ndarray],
                                  covariance_matrix: Union[pd.DataFrame, np.ndarray],
                                  num_points: int = 50,
                                  constraints: Optional[OptimizationConstraints] = None,
                                  risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Generate efficient frontier points.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            num_points: Number of points on the frontier
            constraints: Optimization constraints
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with frontier data
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Find min and max return portfolios
        min_var_result = self.optimize_portfolio(
            expected_returns, covariance_matrix, 'min_variance', constraints, risk_free_rate=risk_free_rate
        )
        
        if not min_var_result.success:
            return {'success': False, 'error': 'Could not find minimum variance portfolio'}
        
        # Find range of returns
        min_return = min_var_result.expected_return
        max_return = np.max(expected_returns)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, num_points)
        
        frontier_data = {
            'returns': [],
            'risks': [],
            'sharpe_ratios': [],
            'weights': [],
            'success': True
        }
        
        for target_return in target_returns:
            target_constraints = OptimizationConstraints(
                min_weight=constraints.min_weight,
                max_weight=constraints.max_weight,
                target_return=target_return,
                long_only=constraints.long_only,
                sum_to_one=constraints.sum_to_one
            )
            
            result = self.optimize_portfolio(
                expected_returns, covariance_matrix, 'min_variance', 
                target_constraints, risk_free_rate=risk_free_rate
            )
            
            if result.success:
                frontier_data['returns'].append(result.expected_return)
                frontier_data['risks'].append(result.expected_risk)
                frontier_data['sharpe_ratios'].append(result.sharpe_ratio)
                frontier_data['weights'].append(result.weights)
        
        return frontier_data
    
    def compare_portfolios(self, 
                          portfolios: Dict[str, pd.Series],
                          expected_returns: Union[pd.Series, np.ndarray],
                          covariance_matrix: Union[pd.DataFrame, np.ndarray],
                          risk_free_rate: float = 0.02) -> pd.DataFrame:
        """
        Compare multiple portfolios on risk-return metrics.
        
        Args:
            portfolios: Dictionary of {name: weights} for each portfolio
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for name, weights in portfolios.items():
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            # Calculate concentration (Herfindahl index)
            concentration = np.sum(weights ** 2)
            
            # Calculate maximum weight
            max_weight = np.max(weights)
            
            results.append({
                'Portfolio': name,
                'Expected Return': portfolio_return,
                'Risk (Volatility)': portfolio_risk,
                'Sharpe Ratio': sharpe_ratio,
                'Concentration': concentration,
                'Max Weight': max_weight,
                'Number of Holdings': np.sum(weights > 1e-6)
            })
        
        return pd.DataFrame(results).set_index('Portfolio')


# Convenience functions for common optimization tasks
def optimize_max_sharpe_portfolio(expected_returns: Union[pd.Series, np.ndarray],
                                 covariance_matrix: Union[pd.DataFrame, np.ndarray],
                                 risk_free_rate: float = 0.02,
                                 min_weight: float = 0.0,
                                 max_weight: float = 1.0) -> OptimizationResult:
    """Quick function to optimize for maximum Sharpe ratio."""
    optimizer = PortfolioOptimizer()
    constraints = OptimizationConstraints(min_weight=min_weight, max_weight=max_weight)
    
    return optimizer.optimize_portfolio(
        expected_returns, covariance_matrix, 'max_sharpe',
        constraints=constraints, risk_free_rate=risk_free_rate
    )

def optimize_risk_parity_portfolio(expected_returns: Union[pd.Series, np.ndarray],
                                  covariance_matrix: Union[pd.DataFrame, np.ndarray],
                                  min_weight: float = 0.01,
                                  max_weight: float = 0.5) -> OptimizationResult:
    """Quick function to optimize for risk parity."""
    optimizer = PortfolioOptimizer()
    constraints = OptimizationConstraints(min_weight=min_weight, max_weight=max_weight)
    
    return optimizer.optimize_portfolio(
        expected_returns, covariance_matrix, 'risk_parity',
        constraints=constraints
    )

def optimize_minimum_variance_portfolio(expected_returns: Union[pd.Series, np.ndarray],
                                       covariance_matrix: Union[pd.DataFrame, np.ndarray],
                                       min_weight: float = 0.0,
                                       max_weight: float = 1.0) -> OptimizationResult:
    """Quick function to optimize for minimum variance."""
    optimizer = PortfolioOptimizer()
    constraints = OptimizationConstraints(min_weight=min_weight, max_weight=max_weight)
    
    return optimizer.optimize_portfolio(
        expected_returns, covariance_matrix, 'min_variance',
        constraints=constraints
    )
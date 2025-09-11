"""
Portfolio Backtesting Framework for optimization strategies.

This module provides comprehensive backtesting capabilities including:
- Walk-forward analysis with rolling windows
- Performance attribution analysis
- Transaction cost modeling
- Multiple rebalancing strategies
- Risk-adjusted performance metrics
- Benchmarking and comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    start_date: str
    end_date: str
    rebalance_frequency: str = 'M'  # D, W, M, Q, Y
    lookback_window: int = 252  # Trading days for estimation window
    min_history: int = 63  # Minimum history required
    transaction_cost: float = 0.001  # 0.1% transaction cost
    benchmark: Optional[str] = None  # Benchmark ticker (e.g., 'SPY')
    initial_capital: float = 1000000.0  # $1M starting capital
    max_weight: float = 0.2  # Maximum asset weight
    min_weight: float = 0.0  # Minimum asset weight
    rebalance_threshold: float = 0.05  # Rebalance if drift > 5%

@dataclass
class PerformanceMetrics:
    """Structure for performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    information_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    skewness: float
    excess_kurtosis: float
    var_95: float
    cvar_95: float
    tracking_error: float = 0.0
    beta: float = 1.0
    alpha: float = 0.0

@dataclass
class BacktestResult:
    """Structure for backtest results."""
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    weights_history: pd.DataFrame
    portfolio_value: pd.Series
    performance_metrics: PerformanceMetrics
    transaction_costs: pd.Series
    turnover: pd.Series
    attribution: Dict[str, Any]
    success: bool
    message: str

class PortfolioBacktester:
    """
    Comprehensive portfolio backtesting framework.
    
    Supports walk-forward analysis, multiple optimization strategies,
    transaction cost modeling, and performance attribution.
    """
    
    def __init__(self):
        self.optimization_methods = {
            'max_sharpe': self._optimize_max_sharpe,
            'min_variance': self._optimize_min_variance,
            'risk_parity': self._optimize_risk_parity,
            'equal_weight': self._optimize_equal_weight,
            '60_40': self._optimize_60_40
        }
    
    def backtest_strategy(self,
                         returns_data: pd.DataFrame,
                         optimization_method: str = 'max_sharpe',
                         config: Optional[BacktestConfig] = None,
                         benchmark_data: Optional[pd.Series] = None,
                         **optimization_kwargs) -> BacktestResult:
        """
        Run comprehensive backtest of optimization strategy.
        
        Args:
            returns_data: DataFrame with asset returns
            optimization_method: Strategy to use for portfolio construction
            config: Backtesting configuration
            benchmark_data: Benchmark returns for comparison
            **optimization_kwargs: Additional parameters for optimization
            
        Returns:
            BacktestResult with complete analysis
        """
        if config is None:
            config = BacktestConfig(
                start_date=returns_data.index[0].strftime('%Y-%m-%d'),
                end_date=returns_data.index[-1].strftime('%Y-%m-%d')
            )
        
        try:
            # Prepare data
            start_date = pd.to_datetime(config.start_date)
            end_date = pd.to_datetime(config.end_date)
            
            # Filter data to backtest period
            backtest_data = returns_data.loc[start_date:end_date]
            
            if len(backtest_data) < config.min_history:
                raise ValueError(f"Insufficient data: {len(backtest_data)} days, need {config.min_history}")
            
            # Generate rebalancing dates
            rebalance_dates = self._generate_rebalance_dates(
                backtest_data.index, config.rebalance_frequency
            )
            
            # Initialize tracking variables
            portfolio_weights = pd.DataFrame(index=backtest_data.index, columns=backtest_data.columns)
            portfolio_returns = pd.Series(index=backtest_data.index, dtype=float)
            transaction_costs = pd.Series(index=backtest_data.index, dtype=float)
            turnover = pd.Series(index=backtest_data.index, dtype=float)
            
            current_weights = None
            
            # Walk-forward optimization
            for i, rebalance_date in enumerate(rebalance_dates):
                # Get historical data for optimization
                history_start = rebalance_date - pd.Timedelta(days=config.lookback_window)
                
                if history_start < backtest_data.index[0]:
                    history_start = backtest_data.index[0]
                
                history_data = backtest_data.loc[history_start:rebalance_date]
                
                if len(history_data) < config.min_history:
                    continue
                
                # Optimize portfolio
                new_weights = self._optimize_portfolio(
                    history_data, optimization_method, config, **optimization_kwargs
                )
                
                if new_weights is None:
                    continue
                
                # Calculate transaction costs and turnover
                if current_weights is not None:
                    trade_size = np.abs(new_weights - current_weights).sum()
                    transaction_cost = trade_size * config.transaction_cost
                    portfolio_turnover = trade_size / 2
                else:
                    transaction_cost = 0.0
                    portfolio_turnover = 0.0
                
                # Update weights until next rebalance
                next_rebalance = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else backtest_data.index[-1]
                
                period_mask = (backtest_data.index >= rebalance_date) & (backtest_data.index <= next_rebalance)
                portfolio_weights.loc[period_mask] = new_weights.values
                
                # Record transaction costs and turnover at rebalance date
                transaction_costs.loc[rebalance_date] = transaction_cost
                turnover.loc[rebalance_date] = portfolio_turnover
                
                current_weights = new_weights.copy()
            
            # Forward-fill weights and calculate returns
            portfolio_weights = portfolio_weights.fillna(method='ffill')
            
            # Calculate portfolio returns
            for date in backtest_data.index:
                if not portfolio_weights.loc[date].isna().all():
                    weights = portfolio_weights.loc[date]
                    daily_return = (backtest_data.loc[date] * weights).sum()
                    
                    # Subtract transaction costs
                    if not pd.isna(transaction_costs.loc[date]) and transaction_costs.loc[date] > 0:
                        daily_return -= transaction_costs.loc[date]
                    
                    portfolio_returns.loc[date] = daily_return
            
            # Calculate portfolio value
            portfolio_value = (1 + portfolio_returns.fillna(0)).cumprod() * config.initial_capital
            
            # Prepare benchmark returns
            if benchmark_data is not None:
                benchmark_returns = benchmark_data.loc[start_date:end_date]
            else:
                benchmark_returns = pd.Series(index=portfolio_returns.index, dtype=float)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                portfolio_returns, benchmark_returns
            )
            
            # Calculate attribution
            attribution = self._calculate_attribution(
                portfolio_returns, portfolio_weights, backtest_data
            )
            
            return BacktestResult(
                portfolio_returns=portfolio_returns.fillna(0),
                benchmark_returns=benchmark_returns.fillna(0),
                weights_history=portfolio_weights.fillna(0),
                portfolio_value=portfolio_value,
                performance_metrics=performance_metrics,
                transaction_costs=transaction_costs.fillna(0),
                turnover=turnover.fillna(0),
                attribution=attribution,
                success=True,
                message="Backtest completed successfully"
            )
            
        except Exception as e:
            return BacktestResult(
                portfolio_returns=pd.Series(),
                benchmark_returns=pd.Series(),
                weights_history=pd.DataFrame(),
                portfolio_value=pd.Series(),
                performance_metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                transaction_costs=pd.Series(),
                turnover=pd.Series(),
                attribution={},
                success=False,
                message=f"Backtest failed: {str(e)}"
            )
    
    def _generate_rebalance_dates(self, date_index: pd.DatetimeIndex, frequency: str) -> List[pd.Timestamp]:
        """Generate rebalancing dates based on frequency."""
        rebalance_dates = []
        
        if frequency == 'D':
            return list(date_index)
        elif frequency == 'W':
            for date in date_index:
                if date.weekday() == 4:  # Friday
                    rebalance_dates.append(date)
        elif frequency == 'M':
            current_month = None
            for date in date_index:
                if current_month != date.month:
                    rebalance_dates.append(date)
                    current_month = date.month
        elif frequency == 'Q':
            current_quarter = None
            for date in date_index:
                quarter = (date.month - 1) // 3 + 1
                if current_quarter != quarter:
                    rebalance_dates.append(date)
                    current_quarter = quarter
        elif frequency == 'Y':
            current_year = None
            for date in date_index:
                if current_year != date.year:
                    rebalance_dates.append(date)
                    current_year = date.year
        
        return rebalance_dates
    
    def _optimize_portfolio(self, history_data: pd.DataFrame, method: str, 
                           config: BacktestConfig, **kwargs) -> Optional[pd.Series]:
        """Optimize portfolio weights using historical data."""
        if method not in self.optimization_methods:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return self.optimization_methods[method](history_data, config, **kwargs)
    
    def _optimize_max_sharpe(self, history_data: pd.DataFrame, config: BacktestConfig, **kwargs) -> pd.Series:
        """Optimize for maximum Sharpe ratio."""
        expected_returns = history_data.mean() * 252
        cov_matrix = history_data.cov() * 252
        
        n_assets = len(expected_returns)
        risk_free_rate = kwargs.get('risk_free_rate', 0.02)
        
        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_risk == 0:
                return 1e6
            return -(portfolio_return - risk_free_rate) / portfolio_risk
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(config.min_weight, config.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        from scipy import optimize
        result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            return pd.Series(np.ones(n_assets) / n_assets, index=expected_returns.index)
    
    def _optimize_min_variance(self, history_data: pd.DataFrame, config: BacktestConfig, **kwargs) -> pd.Series:
        """Optimize for minimum variance."""
        cov_matrix = history_data.cov() * 252
        n_assets = len(cov_matrix)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(config.min_weight, config.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        from scipy import optimize
        result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        else:
            return pd.Series(np.ones(n_assets) / n_assets, index=cov_matrix.index)
    
    def _optimize_risk_parity(self, history_data: pd.DataFrame, config: BacktestConfig, **kwargs) -> pd.Series:
        """Optimize for risk parity."""
        cov_matrix = history_data.cov() * 252
        n_assets = len(cov_matrix)
        
        def objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_risk == 0:
                return 1e6
            
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_risk
            contrib = weights * marginal_contrib
            target_contrib = portfolio_risk / n_assets
            
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(config.min_weight, config.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        from scipy import optimize
        result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        else:
            return pd.Series(np.ones(n_assets) / n_assets, index=cov_matrix.index)
    
    def _optimize_equal_weight(self, history_data: pd.DataFrame, config: BacktestConfig, **kwargs) -> pd.Series:
        """Equal weight portfolio."""
        n_assets = len(history_data.columns)
        weights = np.ones(n_assets) / n_assets
        return pd.Series(weights, index=history_data.columns)
    
    def _optimize_60_40(self, history_data: pd.DataFrame, config: BacktestConfig, **kwargs) -> pd.Series:
        """60/40 stock/bond portfolio (simplified - assumes first asset is stock proxy, second is bond proxy)."""
        if len(history_data.columns) < 2:
            return self._optimize_equal_weight(history_data, config, **kwargs)
        
        weights = pd.Series(0.0, index=history_data.columns)
        weights.iloc[0] = 0.6  # First asset gets 60%
        weights.iloc[1] = 0.4  # Second asset gets 40%
        
        return weights
    
    def _calculate_performance_metrics(self, portfolio_returns: pd.Series, 
                                     benchmark_returns: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        # Basic return metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = portfolio_returns - risk_free_rate / 252
        
        sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else portfolio_returns.std() * np.sqrt(252)
        sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win/loss metrics
        positive_returns = portfolio_returns[portfolio_returns > 0]
        negative_returns = portfolio_returns[portfolio_returns < 0]
        
        win_rate = len(positive_returns) / len(portfolio_returns)
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        # Distribution metrics
        from scipy import stats
        skewness = stats.skew(portfolio_returns.dropna())
        excess_kurtosis = stats.kurtosis(portfolio_returns.dropna())
        
        # VaR metrics
        var_95 = np.percentile(portfolio_returns.dropna(), 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Benchmark comparison metrics
        tracking_error = 0.0
        information_ratio = 0.0
        beta = 1.0
        alpha = 0.0
        
        if len(benchmark_returns) > 0 and not benchmark_returns.isna().all():
            # Align dates
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
            
            if len(aligned_portfolio) > 1:
                excess_portfolio = aligned_portfolio - aligned_benchmark
                tracking_error = excess_portfolio.std() * np.sqrt(252)
                information_ratio = excess_portfolio.mean() / excess_portfolio.std() * np.sqrt(252) if excess_portfolio.std() > 0 else 0
                
                # Calculate beta and alpha
                covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                benchmark_var = np.var(aligned_benchmark)
                beta = covariance / benchmark_var if benchmark_var > 0 else 1.0
                
                portfolio_mean_annual = aligned_portfolio.mean() * 252
                benchmark_mean_annual = aligned_benchmark.mean() * 252
                alpha = portfolio_mean_annual - (risk_free_rate + beta * (benchmark_mean_annual - risk_free_rate))
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            tracking_error=tracking_error,
            beta=beta,
            alpha=alpha
        )
    
    def _calculate_attribution(self, portfolio_returns: pd.Series,
                              weights_history: pd.DataFrame,
                              returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance attribution analysis."""
        attribution = {}
        
        # Asset contribution analysis
        asset_contributions = {}
        for asset in returns_data.columns:
            if asset in weights_history.columns:
                asset_weights = weights_history[asset].fillna(0)
                asset_returns = returns_data[asset]
                
                # Align dates
                aligned_weights, aligned_returns = asset_weights.align(asset_returns, join='inner')
                contribution = (aligned_weights * aligned_returns).sum()
                asset_contributions[asset] = contribution
        
        attribution['asset_contributions'] = asset_contributions
        
        # Allocation vs selection effect (simplified)
        portfolio_return = portfolio_returns.sum()
        equal_weight_return = returns_data.mean(axis=1).sum()
        
        attribution['total_return'] = portfolio_return
        attribution['allocation_effect'] = portfolio_return - equal_weight_return
        attribution['security_selection_effect'] = 0.0  # Placeholder for more advanced attribution
        
        return attribution
    
    def compare_strategies(self, 
                          returns_data: pd.DataFrame,
                          strategies: List[str],
                          config: Optional[BacktestConfig] = None,
                          benchmark_data: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compare multiple optimization strategies.
        
        Args:
            returns_data: Asset returns data
            strategies: List of strategy names to compare
            config: Backtesting configuration
            benchmark_data: Benchmark for comparison
            
        Returns:
            DataFrame with strategy comparison metrics
        """
        results = []
        
        for strategy in strategies:
            backtest_result = self.backtest_strategy(
                returns_data, strategy, config, benchmark_data
            )
            
            if backtest_result.success:
                metrics = backtest_result.performance_metrics
                results.append({
                    'Strategy': strategy,
                    'Total Return': metrics.total_return,
                    'Annualized Return': metrics.annualized_return,
                    'Volatility': metrics.annualized_volatility,
                    'Sharpe Ratio': metrics.sharpe_ratio,
                    'Sortino Ratio': metrics.sortino_ratio,
                    'Max Drawdown': metrics.max_drawdown,
                    'Calmar Ratio': metrics.calmar_ratio,
                    'Win Rate': metrics.win_rate,
                    'Information Ratio': metrics.information_ratio,
                    'Tracking Error': metrics.tracking_error
                })
            else:
                results.append({
                    'Strategy': strategy,
                    'Total Return': np.nan,
                    'Annualized Return': np.nan,
                    'Volatility': np.nan,
                    'Sharpe Ratio': np.nan,
                    'Sortino Ratio': np.nan,
                    'Max Drawdown': np.nan,
                    'Calmar Ratio': np.nan,
                    'Win Rate': np.nan,
                    'Information Ratio': np.nan,
                    'Tracking Error': np.nan
                })
        
        return pd.DataFrame(results).set_index('Strategy')


# Convenience functions
def backtest_max_sharpe_strategy(returns_data: pd.DataFrame,
                                start_date: str,
                                end_date: str,
                                rebalance_frequency: str = 'M') -> BacktestResult:
    """Quick backtest of max Sharpe strategy."""
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency=rebalance_frequency
    )
    
    backtester = PortfolioBacktester()
    return backtester.backtest_strategy(returns_data, 'max_sharpe', config)

def compare_common_strategies(returns_data: pd.DataFrame,
                             start_date: str,
                             end_date: str) -> pd.DataFrame:
    """Compare common portfolio strategies."""
    config = BacktestConfig(start_date=start_date, end_date=end_date)
    strategies = ['equal_weight', 'max_sharpe', 'min_variance', 'risk_parity']
    
    backtester = PortfolioBacktester()
    return backtester.compare_strategies(returns_data, strategies, config)
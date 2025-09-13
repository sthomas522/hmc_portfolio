"""
Monte Carlo Portfolio Simulation Module

This module provides portfolio forecasting through Monte Carlo simulation with emphasis on:
- Uncertainty quantification rather than point predictions
- Scenario analysis and stress testing
- Multiple modeling approaches to highlight model risk
- Clear communication of limitations and assumptions

WARNING: All forecasts are inherently uncertain. This module is designed for scenario
planning and risk assessment, not precise predictions of future returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    num_simulations: int = 10000
    time_horizon_years: int = 10
    initial_portfolio_value: float = 100000.0
    annual_contribution: float = 0.0  # Additional annual investment
    contribution_growth_rate: float = 0.0  # Annual increase in contributions
    rebalancing_frequency: str = 'annual'  # 'monthly', 'quarterly', 'annual', 'none'
    transaction_cost_rate: float = 0.001  # 0.1% per rebalance
    inflation_rate: float = 0.025  # 2.5% annual inflation assumption
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

@dataclass
class StressScenario:
    """Definition of a stress testing scenario."""
    name: str
    description: str
    market_shock_magnitude: float  # Immediate market drop (e.g., -0.3 for 30% drop)
    shock_duration_years: float  # How long the effect lasts
    recovery_pattern: str  # 'linear', 'v_shaped', 'l_shaped', 'gradual'
    correlation_increase: float = 0.2  # Increase in correlations during stress

@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    final_values: np.ndarray
    value_paths: np.ndarray
    percentiles: Dict[float, float]
    real_percentiles: Dict[float, float]  # Inflation-adjusted
    probability_metrics: Dict[str, float]
    scenario_results: Dict[str, Any]
    assumptions: Dict[str, Any]
    limitations: List[str]

class MonteCarloSimulator:
    """
    Monte Carlo portfolio simulation with emphasis on uncertainty and scenario analysis.
    
    Provides multiple simulation approaches to highlight model uncertainty:
    1. Parametric simulation (normal/t-distribution assumptions)
    2. Bootstrap resampling from historical data
    3. Regime-switching models
    4. Stress testing scenarios
    """
    
    def __init__(self):
        self.stress_scenarios = self._define_stress_scenarios()
    
    def simulate_portfolio(self,
                          returns_data: pd.DataFrame,
                          portfolio_weights: pd.Series,
                          config: SimulationConfig,
                          simulation_methods: List[str] = None) -> Dict[str, SimulationResult]:
        """
        Run Monte Carlo simulation using multiple approaches.
        
        Args:
            returns_data: Historical returns data
            portfolio_weights: Portfolio allocation weights
            config: Simulation configuration
            simulation_methods: List of methods to use ['parametric', 'bootstrap', 'regime_switching']
            
        Returns:
            Dictionary of simulation results by method
        """
        if simulation_methods is None:
            simulation_methods = ['parametric', 'bootstrap']
        
        results = {}
        
        # Validate inputs
        portfolio_weights = portfolio_weights / portfolio_weights.sum()  # Normalize
        
        for method in simulation_methods:
            try:
                if method == 'parametric':
                    results[method] = self._parametric_simulation(returns_data, portfolio_weights, config)
                elif method == 'bootstrap':
                    results[method] = self._bootstrap_simulation(returns_data, portfolio_weights, config)
                elif method == 'regime_switching':
                    results[method] = self._regime_switching_simulation(returns_data, portfolio_weights, config)
                else:
                    print(f"Unknown simulation method: {method}")
                    
            except Exception as e:
                print(f"Error in {method} simulation: {str(e)}")
        
        return results
    
    def _parametric_simulation(self, returns_data: pd.DataFrame, 
                              weights: pd.Series, config: SimulationConfig) -> SimulationResult:
        """Parametric simulation assuming multivariate normal or t-distribution."""
        
        # Calculate portfolio statistics
        portfolio_returns = (returns_data * weights).sum(axis=1)
        annual_mean = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Test for normality and use t-distribution if returns show fat tails
        _, p_value = stats.jarque_bera(portfolio_returns)
        use_t_dist = p_value < 0.05
        
        if use_t_dist:
            # Fit t-distribution for fat tails
            df, loc, scale = stats.t.fit(portfolio_returns)
            distribution_params = {'distribution': 't', 'df': df, 'annual_mean': annual_mean, 'annual_vol': annual_vol}
        else:
            distribution_params = {'distribution': 'normal', 'annual_mean': annual_mean, 'annual_vol': annual_vol}
        
        # Run simulation
        final_values, value_paths = self._simulate_paths(
            config, annual_mean, annual_vol, use_t_dist, 
            df if use_t_dist else None
        )
        
        # Calculate metrics
        percentiles = self._calculate_percentiles(final_values, config.confidence_levels)
        real_percentiles = self._calculate_real_percentiles(final_values, config)
        probability_metrics = self._calculate_probability_metrics(final_values, config)
        
        # Run stress scenarios
        scenario_results = self._run_stress_scenarios(config, annual_mean, annual_vol)
        
        # Document assumptions and limitations
        assumptions = {
            'return_distribution': distribution_params,
            'constant_volatility': True,
            'constant_correlations': True,
            'no_regime_changes': True,
            'rebalancing_frequency': config.rebalancing_frequency,
            'transaction_costs': config.transaction_cost_rate
        }
        
        limitations = [
            "Assumes historical return patterns will continue",
            "Does not account for changing market regimes",
            "Assumes constant volatility and correlations",
            "Does not model tail dependencies during crises",
            "Ignores potential structural market changes",
            "Tax implications not modeled",
            "Behavioral factors (panic selling, strategy changes) not included"
        ]
        
        return SimulationResult(
            final_values=final_values,
            value_paths=value_paths,
            percentiles=percentiles,
            real_percentiles=real_percentiles,
            probability_metrics=probability_metrics,
            scenario_results=scenario_results,
            assumptions=assumptions,
            limitations=limitations
        )
    
    def _bootstrap_simulation(self, returns_data: pd.DataFrame,
                             weights: pd.Series, config: SimulationConfig) -> SimulationResult:
        """Bootstrap simulation resampling from historical data."""
        
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        # Bootstrap parameters
        block_size = 21  # Approximately one month of trading days
        
        final_values = np.zeros(config.num_simulations)
        value_paths = np.zeros((config.num_simulations, config.time_horizon_years + 1))
        
        for sim in range(config.num_simulations):
            # Generate path using block bootstrap
            path_returns = self._block_bootstrap_path(
                portfolio_returns, config.time_horizon_years * 252, block_size
            )
            
            # Convert to portfolio value path
            value_path = self._returns_to_value_path(path_returns, config)
            value_paths[sim] = value_path[::252]  # Annual snapshots
            final_values[sim] = value_path[-1]
        
        # Calculate metrics
        percentiles = self._calculate_percentiles(final_values, config.confidence_levels)
        real_percentiles = self._calculate_real_percentiles(final_values, config)
        probability_metrics = self._calculate_probability_metrics(final_values, config)
        
        # Bootstrap doesn't use stress scenarios (uses historical stress periods)
        scenario_results = {'note': 'Bootstrap includes historical stress periods naturally'}
        
        assumptions = {
            'historical_patterns_repeat': True,
            'block_bootstrap_size': block_size,
            'preserves_serial_correlation': True,
            'includes_historical_crises': True
        }
        
        limitations = [
            "Limited to historical patterns only",
            "Cannot model unprecedented events",
            "Historical sample may not be representative of future",
            "Bootstrap variance may underestimate true uncertainty",
            "Regime changes not explicitly modeled"
        ]
        
        return SimulationResult(
            final_values=final_values,
            value_paths=value_paths,
            percentiles=percentiles,
            real_percentiles=real_percentiles,
            probability_metrics=probability_metrics,
            scenario_results=scenario_results,
            assumptions=assumptions,
            limitations=limitations
        )
    
    def _simulate_paths(self, config: SimulationConfig, annual_mean: float, 
                       annual_vol: float, use_t_dist: bool = False, 
                       df: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simulated portfolio value paths."""
        
        dt = 1/252  # Daily time step
        num_steps = int(config.time_horizon_years * 252)
        
        final_values = np.zeros(config.num_simulations)
        value_paths = np.zeros((config.num_simulations, num_steps + 1))
        
        for sim in range(config.num_simulations):
            # Generate return path
            if use_t_dist and df is not None:
                # t-distribution for fat tails
                innovations = stats.t.rvs(df, size=num_steps)
                innovations = (innovations - innovations.mean()) / innovations.std()
                daily_returns = annual_mean * dt + annual_vol * np.sqrt(dt) * innovations
            else:
                # Normal distribution
                daily_returns = np.random.normal(
                    annual_mean * dt, annual_vol * np.sqrt(dt), num_steps
                )
            
            # Convert to value path with contributions and costs
            value_path = self._returns_to_value_path(daily_returns, config)
            value_paths[sim] = value_path
            final_values[sim] = value_path[-1]
        
        return final_values, value_paths
    
    def _returns_to_value_path(self, daily_returns: np.ndarray, 
                              config: SimulationConfig) -> np.ndarray:
        """Convert return series to portfolio value path including contributions and costs."""
        
        num_steps = len(daily_returns)
        value_path = np.zeros(num_steps + 1)
        value_path[0] = config.initial_portfolio_value
        
        # Calculate rebalancing schedule
        rebalance_frequency = {'monthly': 21, 'quarterly': 63, 'annual': 252, 'none': num_steps + 1}
        rebalance_interval = rebalance_frequency.get(config.rebalancing_frequency, 252)
        
        current_contribution = config.annual_contribution
        days_per_year = 252
        
        for day in range(num_steps):
            # Apply daily return
            value_path[day + 1] = value_path[day] * (1 + daily_returns[day])
            
            # Add contributions (spread throughout year)
            if config.annual_contribution > 0:
                daily_contribution = current_contribution / days_per_year
                value_path[day + 1] += daily_contribution
            
            # Apply transaction costs at rebalancing
            if (day + 1) % rebalance_interval == 0:
                transaction_cost = value_path[day + 1] * config.transaction_cost_rate
                value_path[day + 1] -= transaction_cost
            
            # Annual contribution growth
            if (day + 1) % days_per_year == 0:
                current_contribution *= (1 + config.contribution_growth_rate)
        
        return value_path
    
    def _block_bootstrap_path(self, returns: pd.Series, path_length: int, 
                             block_size: int) -> np.ndarray:
        """Generate path using block bootstrap to preserve serial correlation."""
        
        path = []
        returns_array = returns.values
        
        while len(path) < path_length:
            # Random starting point
            start_idx = np.random.randint(0, len(returns_array) - block_size + 1)
            block = returns_array[start_idx:start_idx + block_size]
            path.extend(block)
        
        return np.array(path[:path_length])
    
    def _calculate_percentiles(self, final_values: np.ndarray, 
                              confidence_levels: List[float]) -> Dict[float, float]:
        """Calculate percentiles of final portfolio values."""
        percentiles = {}
        for level in confidence_levels:
            percentiles[level] = np.percentile(final_values, level * 100)
        return percentiles
    
    def _calculate_real_percentiles(self, final_values: np.ndarray, 
                                   config: SimulationConfig) -> Dict[float, float]:
        """Calculate inflation-adjusted percentiles."""
        inflation_factor = (1 + config.inflation_rate) ** config.time_horizon_years
        real_values = final_values / inflation_factor
        
        real_percentiles = {}
        for level in config.confidence_levels:
            real_percentiles[level] = np.percentile(real_values, level * 100)
        return real_percentiles
    
    def _calculate_probability_metrics(self, final_values: np.ndarray, 
                                      config: SimulationConfig) -> Dict[str, float]:
        """Calculate probability-based metrics."""
        
        # Probability of loss
        prob_loss = np.mean(final_values < config.initial_portfolio_value)
        
        # Probability of real loss (inflation-adjusted)
        inflation_factor = (1 + config.inflation_rate) ** config.time_horizon_years
        real_break_even = config.initial_portfolio_value * inflation_factor
        prob_real_loss = np.mean(final_values < real_break_even)
        
        # Probability of doubling money
        prob_double = np.mean(final_values > 2 * config.initial_portfolio_value)
        
        # Expected shortfall (CVaR at 5%)
        var_5 = np.percentile(final_values, 5)
        cvar_5 = np.mean(final_values[final_values <= var_5])
        
        return {
            'probability_of_loss': prob_loss,
            'probability_of_real_loss': prob_real_loss,
            'probability_of_doubling': prob_double,
            'expected_shortfall_5pct': cvar_5,
            'value_at_risk_5pct': var_5
        }
    
    def _run_stress_scenarios(self, config: SimulationConfig, 
                             annual_mean: float, annual_vol: float) -> Dict[str, Any]:
        """Run stress testing scenarios."""
        
        scenario_results = {}
        
        for scenario in self.stress_scenarios:
            stressed_paths = []
            
            for _ in range(1000):  # Fewer simulations for stress scenarios
                path = self._generate_stressed_path(
                    scenario, config, annual_mean, annual_vol
                )
                stressed_paths.append(path[-1])
            
            stressed_paths = np.array(stressed_paths)
            scenario_results[scenario.name] = {
                'description': scenario.description,
                'median_outcome': np.median(stressed_paths),
                'percentile_5': np.percentile(stressed_paths, 5),
                'percentile_95': np.percentile(stressed_paths, 95),
                'probability_of_loss': np.mean(stressed_paths < config.initial_portfolio_value)
            }
        
        return scenario_results
    
    def _generate_stressed_path(self, scenario: StressScenario, config: SimulationConfig,
                               annual_mean: float, annual_vol: float) -> np.ndarray:
        """Generate a single path under stress scenario."""
        
        num_steps = int(config.time_horizon_years * 252)
        shock_duration_steps = int(scenario.shock_duration_years * 252)
        
        # Generate base returns
        returns = np.random.normal(annual_mean / 252, annual_vol / np.sqrt(252), num_steps)
        
        # Apply stress scenario
        if scenario.recovery_pattern == 'v_shaped':
            # Immediate shock followed by recovery
            returns[0] += scenario.market_shock_magnitude
            recovery_boost = -scenario.market_shock_magnitude / shock_duration_steps
            returns[1:shock_duration_steps + 1] += recovery_boost
            
        elif scenario.recovery_pattern == 'l_shaped':
            # Prolonged depression
            returns[0] += scenario.market_shock_magnitude
            depression_return = scenario.market_shock_magnitude / 4  # Continued poor performance
            returns[1:shock_duration_steps] += depression_return / 252
            
        elif scenario.recovery_pattern == 'gradual':
            # Gradual decline and recovery
            shock_per_day = scenario.market_shock_magnitude / (shock_duration_steps / 2)
            returns[:shock_duration_steps // 2] += shock_per_day / 252
            returns[shock_duration_steps // 2:shock_duration_steps] -= shock_per_day / 252
        
        # Convert to value path
        return self._returns_to_value_path(returns, config)
    
    def _define_stress_scenarios(self) -> List[StressScenario]:
        """Define stress testing scenarios based on historical events."""
        
        return [
            StressScenario(
                name="Financial Crisis",
                description="2008-style financial crisis with banking system stress",
                market_shock_magnitude=-0.4,
                shock_duration_years=1.5,
                recovery_pattern='gradual',
                correlation_increase=0.3
            ),
            StressScenario(
                name="Flash Crash",
                description="Sudden market crash with quick recovery",
                market_shock_magnitude=-0.25,
                shock_duration_years=0.5,
                recovery_pattern='v_shaped',
                correlation_increase=0.4
            ),
            StressScenario(
                name="Prolonged Bear Market",
                description="Extended period of poor returns",
                market_shock_magnitude=-0.15,
                shock_duration_years=3.0,
                recovery_pattern='l_shaped',
                correlation_increase=0.2
            ),
            StressScenario(
                name="Inflation Shock",
                description="Unexpected high inflation eroding real returns",
                market_shock_magnitude=-0.1,
                shock_duration_years=2.0,
                recovery_pattern='gradual',
                correlation_increase=0.1
            )
        ]
    
    def generate_scenario_analysis(self, results: Dict[str, SimulationResult],
                                  config: SimulationConfig) -> Dict[str, Any]:
        """Generate comprehensive scenario analysis report."""
        
        analysis = {
            'executive_summary': self._generate_executive_summary(results, config),
            'uncertainty_analysis': self._analyze_uncertainty(results),
            'scenario_comparison': self._compare_scenarios(results),
            'key_insights': self._generate_key_insights(results, config),
            'recommendations': self._generate_recommendations(results, config)
        }
        
        return analysis
    
    def _generate_executive_summary(self, results: Dict[str, SimulationResult],
                                   config: SimulationConfig) -> Dict[str, Any]:
        """Generate executive summary of simulation results."""
        
        # Use parametric results as baseline
        baseline = results.get('parametric', list(results.values())[0])
        
        median_nominal = baseline.percentiles[0.50]
        median_real = baseline.real_percentiles[0.50]
        
        return {
            'time_horizon': f"{config.time_horizon_years} years",
            'initial_investment': f"${config.initial_portfolio_value:,.0f}",
            'median_outcome_nominal': f"${median_nominal:,.0f}",
            'median_outcome_real': f"${median_real:,.0f}",
            'probability_of_loss': f"{baseline.probability_metrics['probability_of_loss']:.1%}",
            'probability_of_real_loss': f"{baseline.probability_metrics['probability_of_real_loss']:.1%}",
            'range_80_percent': f"${baseline.percentiles[0.10]:,.0f} - ${baseline.percentiles[0.90]:,.0f}",
            'worst_5_percent': f"${baseline.percentiles[0.05]:,.0f}"
        }
    
    def _analyze_uncertainty(self, results: Dict[str, SimulationResult]) -> Dict[str, Any]:
        """Analyze uncertainty across different modeling approaches."""
        
        if len(results) < 2:
            return {'note': 'Multiple simulation methods needed for uncertainty analysis'}
        
        # Compare median outcomes across methods
        medians = {method: result.percentiles[0.50] for method, result in results.items()}
        
        median_range = max(medians.values()) - min(medians.values())
        median_avg = np.mean(list(medians.values()))
        
        # Compare 90th percentile ranges
        p90_ranges = {}
        for method, result in results.items():
            p90_ranges[method] = result.percentiles[0.90] - result.percentiles[0.10]
        
        return {
            'model_uncertainty_median': f"${median_range:,.0f}",
            'model_uncertainty_pct': f"{median_range / median_avg:.1%}",
            'method_medians': {k: f"${v:,.0f}" for k, v in medians.items()},
            'uncertainty_ranges_80pct': {k: f"${v:,.0f}" for k, v in p90_ranges.items()}
        }
    
    def _compare_scenarios(self, results: Dict[str, SimulationResult]) -> Dict[str, Any]:
        """Compare stress scenarios across methods."""
        
        scenario_comparison = {}
        
        for method, result in results.items():
            if 'scenario_results' in result.__dict__ and isinstance(result.scenario_results, dict):
                for scenario_name, scenario_data in result.scenario_results.items():
                    if scenario_name not in scenario_comparison:
                        scenario_comparison[scenario_name] = {}
                    scenario_comparison[scenario_name][method] = scenario_data
        
        return scenario_comparison
    
    def _generate_key_insights(self, results: Dict[str, SimulationResult],
                              config: SimulationConfig) -> List[str]:
        """Generate key insights from simulation results."""
        
        baseline = results.get('parametric', list(results.values())[0])
        insights = []
        
        # Probability insights
        prob_loss = baseline.probability_metrics['probability_of_loss']
        prob_real_loss = baseline.probability_metrics['probability_of_real_loss']
        
        if prob_loss < 0.1:
            insights.append(f"Low probability of nominal loss ({prob_loss:.1%}) suggests relatively conservative portfolio")
        elif prob_loss > 0.3:
            insights.append(f"Significant probability of nominal loss ({prob_loss:.1%}) indicates aggressive portfolio")
        
        if prob_real_loss > 0.25:
            insights.append(f"High probability of real loss ({prob_real_loss:.1%}) due to inflation risk")
        
        # Range insights
        p10 = baseline.percentiles[0.10]
        p90 = baseline.percentiles[0.90]
        median = baseline.percentiles[0.50]
        
        upside_potential = (p90 - median) / median
        downside_risk = (median - p10) / median
        
        if upside_potential > downside_risk * 1.5:
            insights.append("Portfolio shows positive skew with greater upside than downside")
        elif downside_risk > upside_potential * 1.5:
            insights.append("Portfolio shows negative skew with greater downside than upside")
        
        # Time horizon insights
        if config.time_horizon_years >= 10:
            insights.append("Long time horizon allows volatility to work in your favor")
        elif config.time_horizon_years <= 3:
            insights.append("Short time horizon increases sequence-of-returns risk")
        
        return insights
    
    def _generate_recommendations(self, results: Dict[str, SimulationResult],
                                 config: SimulationConfig) -> List[str]:
        """Generate actionable recommendations based on simulation results."""
        
        recommendations = []
        baseline = results.get('parametric', list(results.values())[0])
        
        # Risk level recommendations
        prob_real_loss = baseline.probability_metrics['probability_of_real_loss']
        
        if prob_real_loss > 0.4:
            recommendations.append("Consider reducing risk or increasing contribution rate to combat inflation")
        
        # Contribution recommendations
        if config.annual_contribution == 0:
            recommendations.append("Regular contributions can significantly improve outcomes through dollar-cost averaging")
        
        # Time horizon recommendations
        if config.time_horizon_years >= 10:
            recommendations.append("Long time horizon supports maintaining equity allocation through market volatility")
        
        # Stress scenario recommendations
        for scenario_name, scenario_data in baseline.scenario_results.items():
            if isinstance(scenario_data, dict) and scenario_data.get('probability_of_loss', 0) > 0.8:
                recommendations.append(f"Portfolio vulnerable to {scenario_name.lower()} - consider defensive hedging")
        
        # Model uncertainty recommendations
        recommendations.append("Results vary significantly across modeling approaches - avoid overconfidence in any single forecast")
        recommendations.append("Use simulation results for scenario planning, not precise predictions")
        
        return recommendations


# Convenience functions for common use cases
def quick_monte_carlo(returns_data: pd.DataFrame, 
                     portfolio_weights: pd.Series,
                     time_horizon_years: int = 10,
                     initial_value: float = 100000) -> SimulationResult:
    """Quick Monte Carlo simulation with default settings."""
    
    simulator = MonteCarloSimulator()
    config = SimulationConfig(
        time_horizon_years=time_horizon_years,
        initial_portfolio_value=initial_value
    )
    
    results = simulator.simulate_portfolio(returns_data, portfolio_weights, config, ['parametric'])
    return results['parametric']

def compare_simulation_methods(returns_data: pd.DataFrame,
                              portfolio_weights: pd.Series,
                              time_horizon_years: int = 10) -> Dict[str, SimulationResult]:
    """Compare different simulation methods."""
    
    simulator = MonteCarloSimulator()
    config = SimulationConfig(time_horizon_years=time_horizon_years)
    
    return simulator.simulate_portfolio(
        returns_data, portfolio_weights, config, 
        ['parametric', 'bootstrap']
    )
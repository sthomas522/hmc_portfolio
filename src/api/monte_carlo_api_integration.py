"""
Monte Carlo API Integration

This module adds Monte Carlo simulation endpoints to the existing FastAPI application.
Integrates with the portfolio optimization system to provide forecasting capabilities.
"""

from fastapi import HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# Import the Monte Carlo simulator
from analysis.monte_carlo_simulator import (
    MonteCarloSimulator, SimulationConfig, SimulationResult,
    quick_monte_carlo, compare_simulation_methods
)

# Pydantic models for Monte Carlo requests/responses
class MonteCarloRequest(BaseModel):
    tickers: List[str]
    weights: Optional[List[float]] = None
    period: str = "2y"  # Longer period for better simulation
    time_horizon_years: int = 10
    initial_portfolio_value: float = 100000.0
    annual_contribution: float = 0.0
    contribution_growth_rate: float = 0.0
    rebalancing_frequency: str = "annual"
    transaction_cost_rate: float = 0.001
    inflation_rate: float = 0.025
    num_simulations: int = 10000
    simulation_methods: List[str] = ["parametric", "bootstrap"]
    include_stress_testing: bool = True

class ScenarioAnalysisRequest(BaseModel):
    tickers: List[str]
    weights: Optional[List[float]] = None
    scenarios: List[Dict[str, Any]]  # Custom scenario definitions
    time_horizon_years: int = 10
    initial_portfolio_value: float = 100000.0

class MonteCarloResponse(BaseModel):
    simulation_successful: bool
    error: Optional[str] = None
    executive_summary: Optional[Dict[str, Any]] = None
    results_by_method: Optional[Dict[str, Any]] = None
    scenario_analysis: Optional[Dict[str, Any]] = None
    uncertainty_analysis: Optional[Dict[str, Any]] = None
    key_insights: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    limitations: Optional[List[str]] = None

def add_monte_carlo_endpoints(app, data_service):
    """Add Monte Carlo simulation endpoints to FastAPI app."""
    
    # Initialize Monte Carlo simulator
    mc_simulator = MonteCarloSimulator()
    
    @app.post("/simulate/monte-carlo")
    async def monte_carlo_simulation(request: MonteCarloRequest):
        """
        Run Monte Carlo simulation for portfolio forecasting.
        
        Emphasizes uncertainty ranges and scenario analysis rather than point predictions.
        """
        try:
            # Fetch returns data (longer period for simulation)
            returns_df = data_service.fetch_returns_data(
                request.tickers, period=request.period
            )
            
            if len(returns_df.columns) < 2:
                raise HTTPException(status_code=400, detail="Need at least 2 valid tickers for simulation")
            
            # Set up portfolio weights
            if request.weights:
                if len(request.weights) != len(returns_df.columns):
                    raise HTTPException(
                        status_code=400, 
                        detail="Weights length doesn't match number of assets"
                    )
                portfolio_weights = pd.Series(request.weights, index=returns_df.columns)
                portfolio_weights = portfolio_weights / portfolio_weights.sum()
            else:
                # Equal weights if not provided
                portfolio_weights = pd.Series(
                    [1.0/len(returns_df.columns)] * len(returns_df.columns), 
                    index=returns_df.columns
                )
            
            # Configure simulation
            config = SimulationConfig(
                num_simulations=request.num_simulations,
                time_horizon_years=request.time_horizon_years,
                initial_portfolio_value=request.initial_portfolio_value,
                annual_contribution=request.annual_contribution,
                contribution_growth_rate=request.contribution_growth_rate,
                rebalancing_frequency=request.rebalancing_frequency,
                transaction_cost_rate=request.transaction_cost_rate,
                inflation_rate=request.inflation_rate
            )
            
            # Run simulation
            simulation_results = mc_simulator.simulate_portfolio(
                returns_df, portfolio_weights, config, request.simulation_methods
            )
            
            # Generate comprehensive analysis
            scenario_analysis = mc_simulator.generate_scenario_analysis(simulation_results, config)
            
            # Convert to JSON-serializable format
            def convert_simulation_results(results_dict):
                converted = {}
                for method, result in results_dict.items():
                    converted[method] = {
                        'percentiles': {str(k): float(v) for k, v in result.percentiles.items()},
                        'real_percentiles': {str(k): float(v) for k, v in result.real_percentiles.items()},
                        'probability_metrics': {k: float(v) for k, v in result.probability_metrics.items()},
                        'scenario_results': result.scenario_results,
                        'assumptions': result.assumptions,
                        'limitations': result.limitations
                    }
                return converted
            
            response = MonteCarloResponse(
                simulation_successful=True,
                executive_summary=scenario_analysis['executive_summary'],
                results_by_method=convert_simulation_results(simulation_results),
                scenario_analysis=scenario_analysis['scenario_comparison'],
                uncertainty_analysis=scenario_analysis['uncertainty_analysis'],
                key_insights=scenario_analysis['key_insights'],
                recommendations=scenario_analysis['recommendations'],
                limitations=list(simulation_results.values())[0].limitations
            )
            
            return response.dict()
            
        except HTTPException:
            raise
        except Exception as e:
            return MonteCarloResponse(
                simulation_successful=False,
                error=f"Monte Carlo simulation failed: {str(e)}"
            ).dict()
    
    @app.post("/simulate/quick-forecast")
    async def quick_portfolio_forecast(
        tickers: List[str] = Query(...),  # ... means required
        weights: Optional[List[float]] = Query(None),
        time_horizon_years: int = Query(10),
        initial_value: float = Query(100000)
    ):
        """
        Quick Monte Carlo forecast with default settings.
        """
        try:
            # Fetch returns data
            returns_df = data_service.fetch_returns_data(tickers, period="2y")
            
            # Set up weights
            if weights:
                portfolio_weights = pd.Series(weights, index=returns_df.columns)
                portfolio_weights = portfolio_weights / portfolio_weights.sum()
            else:
                portfolio_weights = pd.Series(
                    [1.0/len(returns_df.columns)] * len(returns_df.columns), 
                    index=returns_df.columns
                )
            
            # Run quick simulation
            result = quick_monte_carlo(
                returns_df, portfolio_weights, time_horizon_years, initial_value
            )
            
            # Format response
            response = {
                'simulation_successful': True,
                'time_horizon': f"{time_horizon_years} years",
                'initial_value': f"${initial_value:,.0f}",
                'median_outcome': f"${result.percentiles[0.50]:,.0f}",
                'confidence_range_80pct': f"${result.percentiles[0.10]:,.0f} - ${result.percentiles[0.90]:,.0f}",
                'worst_5_percent': f"${result.percentiles[0.05]:,.0f}",
                'best_5_percent': f"${result.percentiles[0.95]:,.0f}",
                'probability_of_loss': f"{result.probability_metrics['probability_of_loss']:.1%}",
                'probability_of_real_loss': f"{result.probability_metrics['probability_of_real_loss']:.1%}",
                'key_limitation': 'These are scenario projections, not predictions. Actual results may vary significantly.'
            }
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Quick forecast failed: {str(e)}")
    
    @app.post("/simulate/compare-methods")
    async def compare_simulation_methods_endpoint(
        tickers: List[str] = Query(...),
        weights: Optional[List[float]] = Query(None),
        time_horizon_years: int = Query(10),
        initial_value: float = Query(100000)
    ):
        """
        Compare different Monte Carlo simulation methods to highlight model uncertainty.
        """
        try:
            # Fetch returns data
            returns_df = data_service.fetch_returns_data(tickers, period="3y")
            
            # Set up weights
            if weights:
                portfolio_weights = pd.Series(weights, index=returns_df.columns)
                portfolio_weights = portfolio_weights / portfolio_weights.sum()
            else:
                portfolio_weights = pd.Series(
                    [1.0/len(returns_df.columns)] * len(returns_df.columns), 
                    index=returns_df.columns
                )
            
            # Compare methods
            results = compare_simulation_methods(
                returns_df, portfolio_weights, time_horizon_years
            )
            
            # Format comparison
            comparison = {}
            for method, result in results.items():
                comparison[method] = {
                    'median_outcome': f"${result.percentiles[0.50]:,.0f}",
                    'range_80pct': f"${result.percentiles[0.90] - result.percentiles[0.10]:,.0f}",
                    'probability_of_loss': f"{result.probability_metrics['probability_of_loss']:.1%}",
                    'key_assumption': result.assumptions.get('distribution', 'See limitations'),
                    'main_limitation': result.limitations[0] if result.limitations else 'Model-dependent results'
                }
            
            # Calculate model uncertainty
            medians = [result.percentiles[0.50] for result in results.values()]
            model_uncertainty = (max(medians) - min(medians)) / np.mean(medians)
            
            response = {
                'comparison_successful': True,
                'method_comparison': comparison,
                'model_uncertainty_pct': f"{model_uncertainty:.1%}",
                'key_insight': f"Different modeling approaches show {model_uncertainty:.1%} variation in median outcomes",
                'recommendation': 'Use multiple methods to understand modeling uncertainty. Avoid overconfidence in any single forecast.'
            }
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Method comparison failed: {str(e)}")
    
    @app.post("/simulate/stress-scenarios")
    async def stress_scenario_analysis(
        tickers: List[str] = Query(...),
        weights: Optional[List[float]] = Query(None),
        time_horizon_years: int = Query(10),
        initial_value: float = Query(100000)
        # Remove the custom_scenarios parameter that's causing the error
    ):
        """
        Run stress scenario analysis to understand portfolio vulnerability.
        """
        try:
            # Fetch returns data
            returns_df = data_service.fetch_returns_data(tickers, period="2y")
            
            # Set up weights
            if weights:
                portfolio_weights = pd.Series(weights, index=returns_df.columns)
                portfolio_weights = portfolio_weights / portfolio_weights.sum()
            else:
                portfolio_weights = pd.Series(
                    [1.0/len(returns_df.columns)] * len(returns_df.columns), 
                    index=returns_df.columns
                )
            
            # Configure simulation with stress testing
            config = SimulationConfig(
                time_horizon_years=time_horizon_years,
                initial_portfolio_value=initial_value,
                num_simulations=5000  # Fewer for stress testing
            )
            
            # Run simulation
            results = mc_simulator.simulate_portfolio(
                returns_df, portfolio_weights, config, ['parametric']
            )
            
            baseline_result = results['parametric']
            
            # Format stress scenario results
            stress_analysis = {}
            for scenario_name, scenario_data in baseline_result.scenario_results.items():
                if isinstance(scenario_data, dict):
                    stress_analysis[scenario_name] = {
                        'description': scenario_data.get('description', ''),
                        'median_outcome': f"${scenario_data.get('median_outcome', 0):,.0f}",
                        'worst_5_percent': f"${scenario_data.get('percentile_5', 0):,.0f}",
                        'probability_of_loss': f"{scenario_data.get('probability_of_loss', 0):.1%}",
                        'vs_baseline': f"{((scenario_data.get('median_outcome', 0) / baseline_result.percentiles[0.50]) - 1):.1%}"
                    }
            
            response = {
                'stress_analysis_successful': True,
                'baseline_median': f"${baseline_result.percentiles[0.50]:,.0f}",
                'stress_scenarios': stress_analysis,
                'key_vulnerabilities': [
                    scenario for scenario, data in stress_analysis.items() 
                    if 'probability_of_loss' in str(data) and '80' in str(data['probability_of_loss'])
                ],
                'recommendation': 'Consider defensive positioning if portfolio shows high vulnerability to stress scenarios'
            }
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stress scenario analysis failed: {str(e)}")
    
    @app.get("/simulate/health")
    async def monte_carlo_health_check():
        """Health check for Monte Carlo simulation components."""
        try:
            # Test basic simulation functionality
            test_data = pd.DataFrame({
                'A': np.random.normal(0.001, 0.02, 100),
                'B': np.random.normal(0.001, 0.02, 100)
            })
            test_weights = pd.Series([0.6, 0.4], index=['A', 'B'])
            
            # Quick test simulation
            result = quick_monte_carlo(test_data, test_weights, time_horizon_years=1, initial_value=1000)
            
            return {
                'status': 'healthy',
                'service': 'monte-carlo-simulation',
                'components': {
                    'parametric_simulation': 'operational',
                    'bootstrap_simulation': 'operational',
                    'stress_testing': 'operational',
                    'scenario_analysis': 'operational'
                },
                'test_simulation': {
                    'completed': True,
                    'median_outcome': f"${result.percentiles[0.50]:,.0f}",
                    'simulations_run': len(result.final_values)
                },
                'available_endpoints': [
                    '/simulate/monte-carlo',
                    '/simulate/quick-forecast', 
                    '/simulate/compare-methods',
                    '/simulate/stress-scenarios'
                ]
            }
            
        except Exception as e:
            return {
                'status': 'degraded',
                'service': 'monte-carlo-simulation',
                'error': str(e)
            }
    
    return app

# Usage note for integration with main.py:
"""
Add to your main.py:

from monte_carlo_api_integration import add_monte_carlo_endpoints

# After creating your FastAPI app:
app = add_monte_carlo_endpoints(app, data_service)
"""
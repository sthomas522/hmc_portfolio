# HMC Portfolio Analyzer

Advanced retirement portfolio analysis tool implementing QR decomposition optimization from modern Monte Carlo research. Provides institutional-grade risk analysis for high-dimensional portfolios that traditional tools cannot handle effectively.

## Key Features

### Core Analysis
- **High-dimensional correlation analysis** with eigenvalue decomposition
- **Effective rank calculation** revealing true diversification (not just asset count)
- **QR decomposition optimization** providing 100x+ numerical stability improvements
- **Advanced risk metrics** including VaR, CVaR, and maximum drawdown
- **Monte Carlo simulations** optimized for large portfolios

### Retirement-Specific Insights
- **Concentration risk warnings** for portfolios with 50+ assets
- **Long-term wealth projections** with correlation-adjusted uncertainty
- **Diversification effectiveness** analysis showing actual vs. apparent diversification
- **Numerical stability** for reliable 30-year projections

## Installation

### Prerequisites
- Python 3.9+
- uv package manager (recommended) or pip

### Setup
```bash
# Clone the repository
git clone https://github.com/sthomas522/hmc_portfolio.git
cd hmc_portfolio

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

```python
from hmc_portfolio.optimized_analyzer import OptimizedPortfolioAnalyzer

# Your portfolio tickers
tickers = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'VTI', 'BND', 'VEA', 'VWO']
weights = [0.15, 0.15, 0.10, 0.10, 0.25, 0.15, 0.05, 0.05]  # Optional

# Create analyzer
analyzer = OptimizedPortfolioAnalyzer(tickers, weights, start_date='2020-01-01')

# Run analysis
if analyzer.fetch_data():
    analyzer.generate_comprehensive_report()
    analyzer.plot_optimization_analysis(save_plots=True)
```

## Example Output

### Portfolio Analysis
- **Effective rank**: 7.57 out of 60 assets (reveals actual diversification)
- **Condition number improvement**: 202x (numerical stability gain)
- **Concentration ratio**: 0.126 (warns of hidden concentration risk)
- **30-year median projection**: 100.8x initial investment

### Risk Warnings
- Identifies when portfolios have fewer independent risk factors than apparent
- Warns about numerical instability in correlation matrices
- Highlights concentration risk in seemingly diversified portfolios

## Why This Approach Matters

Traditional portfolio tools fail with large portfolios because:
- **Correlation matrices become numerically unstable** (high condition numbers)
- **Asset count doesn't equal diversification** (many stocks, few risk factors)
- **Standard Monte Carlo breaks down** in high dimensions

This tool uses QR decomposition to:
- **Transform correlation matrices** to numerically stable form
- **Reveal true risk factors** through eigenvalue analysis
- **Enable reliable projections** for retirement planning

## Technical Foundation

Based on research in modern Monte Carlo methods and high-dimensional portfolio optimization:
- QR decomposition for numerical stability
- Eigenvalue analysis for true diversification measurement
- Correlation-adjusted Monte Carlo for realistic projections
- Designed for portfolios with 50+ individual securities

## Testing

```bash
# Run comprehensive tests
uv run python test_optimized_example.py

# Tests medium (25), large (60), and mega (120) portfolios
# Demonstrates QR optimization benefits at scale
```

## Project Structure

```
hmc_portfolio/
├── src/hmc_portfolio/
│   ├── __init__.py
│   └── optimized_analyzer.py    # Main analysis tool
├── examples/                    # Usage examples
├── archive/                     # Development history
├── test_optimized_example.py    # Comprehensive tests
└── README.md                    # This file
```

## Limitations

- **Educational/research tool**: Not investment advice
- **Historical data**: Past performance doesn't predict future results
- **US markets focused**: Primarily tested with US equities and ETFs
- **No real-time data**: Uses daily close prices

## Mathematical Background

The tool implements several advanced techniques:

**QR Decomposition**: Transforms correlation matrices from ill-conditioned (condition numbers >10,000) to well-conditioned (≈1) for stable computation.

**Effective Rank**: Calculates the true number of independent risk factors using eigenvalue decomposition, often revealing that 100+ asset portfolios have fewer than 10 independent sources of risk.

**Correlation-Adjusted Monte Carlo**: Uses portfolio-level variance that properly accounts for correlation structure rather than naive independent simulations.

## Contributing

This tool bridges academic research with practical portfolio analysis. Contributions welcome for:
- Additional risk metrics
- International market support
- Performance optimizations
- Visualization improvements

## Disclaimer

This software is for educational and research purposes. Not financial advice. Consult qualified financial advisors before making investment decisions. Past performance does not guarantee future results.

## License

MIT License - see LICENSE file for details.

---

**Built with modern Monte Carlo methods for robust retirement portfolio analysis.**
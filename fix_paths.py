import os

# Fix test file paths
test_content = '''# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))'''

with open('test_optimization_backend.py', 'r') as f:
    content = f.read()

# Fix sys.path
content = content.replace(
    '''sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))''',
    '''sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))'''
)

# Fix pandas
content = content.replace('na_last=True', '')

with open('test_optimization_backend.py', 'w') as f:
    f.write(content)

# Fix integration imports
integration_path = 'src/analysis/optimization_integration.py'
if os.path.exists(integration_path):
    with open(integration_path, 'r') as f:
        content = f.read()
    
    # Remove the manual path manipulation
    content = content.replace('''# Add the analysis module to the path for imports
current_dir = os.path.dirname(__file__)
analysis_dir = os.path.abspath(os.path.join(current_dir, '..', 'analysis'))
sys.path.insert(0, analysis_dir)

# Import the existing analyzer and new optimization components
from portfolio_analyzer import EnhancedPortfolioAnalyzer, PortfolioCorrelationAnalyzer
from portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints, OptimizationResult
from portfolio_backtester import PortfolioBacktester, BacktestConfig, BacktestResult''', 
'''# Import the existing analyzer and new optimization components
from .portfolio_analyzer import EnhancedPortfolioAnalyzer, PortfolioCorrelationAnalyzer
from .portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints, OptimizationResult
from .portfolio_backtester import PortfolioBacktester, BacktestConfig, BacktestResult''')
    
    with open(integration_path, 'w') as f:
        f.write(content)

print("Fixed all import paths and compatibility issues")
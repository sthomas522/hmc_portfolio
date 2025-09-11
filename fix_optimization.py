import os

# Fix the integration imports
integration_file = 'analysis/optimization_integration.py'
if os.path.exists(integration_file):
    with open(integration_file, 'r') as f:
        content = f.read()
    
    content = content.replace(
        'from portfolio_analyzer import',
        'from .portfolio_analyzer import'
    ).replace(
        'from portfolio_optimizer import',
        'from .portfolio_optimizer import'
    ).replace(
        'from portfolio_backtester import',
        'from .portfolio_backtester import'
    )
    
    with open(integration_file, 'w') as f:
        f.write(content)
    
    print("Fixed integration imports")

# Fix the test file
test_file = 'test_optimization_backend.py'
if os.path.exists(test_file):
    with open(test_file, 'r') as f:
        content = f.read()
    
    content = content.replace(
        'na_last=True',
        ''
    )
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    print("Fixed test compatibility")
"""
Test Stan model creation and compilation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hmc_portfolio import RetirementPortfolioAnalyzer

def test_stan_model_creation():
    """Test that the Stan model can be created and compiled"""
    print("🔍 Testing Stan model creation...")
    
    # Create analyzer
    analyzer = RetirementPortfolioAnalyzer(['AAPL', 'MSFT'], start_date='2023-01-01')
    
    # Test Stan model generation
    print("📝 Generating Stan model code...")
    try:
        stan_code = analyzer.create_stan_model()
        print(f"✅ Stan code generated ({len(stan_code)} characters)")
        print("\n📄 Stan model preview:")
        print("-" * 50)
        print(stan_code[:300] + "..." if len(stan_code) > 300 else stan_code)
        print("-" * 50)
        
        # Save to file for inspection
        with open('test_portfolio_model.stan', 'w') as f:
            f.write(stan_code)
        print("✅ Stan model saved to 'test_portfolio_model.stan'")
        
        return True
        
    except Exception as e:
        print(f"❌ Stan model generation failed: {e}")
        return False

def test_cmdstan_availability():
    """Test CmdStanPy availability"""
    print("\n🔍 Testing CmdStanPy installation...")
    
    try:
        import cmdstanpy
        print(f"✅ CmdStanPy imported (version: {cmdstanpy.__version__})")
        
        try:
            cmdstan_path = cmdstanpy.cmdstan_path()
            print(f"✅ CmdStan found at: {cmdstan_path}")
            return True
        except Exception as e:
            print(f"❌ CmdStan not found: {e}")
            print("💡 Install with: python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'")
            return False
            
    except ImportError:
        print("❌ CmdStanPy not installed")
        print("💡 Install with: uv add --optional bayesian cmdstanpy")
        return False

def test_model_compilation():
    """Test that Stan model can be compiled"""
    print("\n🔍 Testing Stan model compilation...")
    
    try:
        import cmdstanpy
        
        # Use the test model file
        if not os.path.exists('test_portfolio_model.stan'):
            print("❌ test_portfolio_model.stan not found")
            return False
        
        print("🔨 Compiling Stan model...")
        model = cmdstanpy.CmdStanModel(stan_file='test_portfolio_model.stan')
        print("✅ Stan model compiled successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Stan model compilation failed: {e}")
        return False

def main():
    """Run all Stan tests"""
    print("🚀 Stan Model Testing")
    print("=" * 50)
    
    tests = [
        ("Stan Model Creation", test_stan_model_creation),
        ("CmdStanPy Availability", test_cmdstan_availability),
        ("Model Compilation", test_model_compilation)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        success = test_func()
        results.append((name, success))
    
    print("\n" + "=" * 50)
    print("📊 STAN TEST SUMMARY:")
    
    passed = 0
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Stan setup is working correctly!")
    else:
        print("❌ Some Stan tests failed.")
        if passed >= 1:  # Model creation works
            print("💡 You can still use the traditional analysis features")

if __name__ == "__main__":
    main()
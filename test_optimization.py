"""
Quick test script for optimization - runs 5 trials quickly
Use this to verify the optimization pipeline works before running full optimization.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Import the optimizer
from optimize_prophet import ProphetOptimizer

def quick_test():
    """Run a quick optimization test with minimal trials"""
    print("="*80)
    print("QUICK OPTIMIZATION TEST")
    print("="*80)
    print("This is a quick test with 5 trials to verify the setup.")
    print("For full optimization, run: python optimize_prophet.py")
    print("="*80)
    
    try:
        # Create optimizer with minimal settings
        optimizer = ProphetOptimizer(
            data_path='data.csv',
            n_trials=5,   # Very few trials for quick test
            n_folds=2     # Fewer folds for speed
        )
        
        print("\n✓ Optimizer created successfully")
        print("✓ Data loaded successfully")
        print(f"✓ Training data: {len(optimizer.train_target)} weeks")
        
        # Run optimization
        print("\nRunning optimization with 5 trials...")
        best_params, best_score = optimizer.optimize()
        
        print("\n" + "="*80)
        print("QUICK TEST COMPLETED!")
        print("="*80)
        print(f"\nBest MAPE from 5 trials: {best_score:.2f}%")
        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print("\n" + "="*80)
        print("✓ Optimization pipeline is working correctly!")
        print("\nTo run full optimization (recommended):")
        print("  python optimize_prophet.py")
        print("\nThis will run 30 trials and give much better results.")
        print("="*80)
        
        return True
        
    except FileNotFoundError:
        print("\n❌ Error: data.csv not found!")
        print("Please run: python generate_data.py")
        return False
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)

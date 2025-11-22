"""
Run the complete time series forecasting pipeline.
This script generates data and runs the forecasting model.
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and display its output"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run the complete pipeline"""
    print("="*60)
    print("TIME SERIES FORECASTING PIPELINE")
    print("="*60)
    print("This script will:")
    print("1. Generate synthetic data (generate_data.py)")
    print("2. Train model and make predictions (forecast_model.py)")
    print("="*60)
    
    # Step 1: Generate data
    if not run_script("generate_data.py"):
        print("\nFailed to generate data. Exiting.")
        sys.exit(1)
    
    # Check if data file was created
    if not os.path.exists("data.csv"):
        print("\nError: data.csv was not created. Exiting.")
        sys.exit(1)
    
    print("\n✓ Data generation completed successfully!")
    
    # Step 2: Run forecasting model
    if not run_script("forecast_model.py"):
        print("\nFailed to run forecasting model. Exiting.")
        sys.exit(1)
    
    print("\n✓ Forecasting completed successfully!")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    
    files = [
        ("data.csv", "Generated time series data"),
        ("predictions.csv", "Model predictions for 104 weeks"),
        ("forecast_results.png", "Visualization of results")
    ]
    
    for filename, description in files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✓ {filename} ({size:,} bytes) - {description}")
        else:
            print(f"  ✗ {filename} - {description} (NOT FOUND)")
    
    print("\n" + "="*60)
    print("You can now review the predictions and visualizations!")
    print("="*60)

if __name__ == "__main__":
    main()

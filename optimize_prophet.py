"""
Hyperparameter Optimization for Prophet Model using Optuna and Backtesting.

This script:
1. Uses Optuna to search for optimal Prophet hyperparameters
2. Evaluates each configuration using backtesting (historical validation)
3. Finds the best model configuration based on MAPE
4. Visualizes optimization results
5. Saves the best model and parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)

from darts import TimeSeries
from darts.models import Prophet
from darts.metrics import mape, rmse, mae
from darts.dataprocessing.transformers import Scaler
import json
from datetime import datetime

# Optuna logging
optuna.logging.set_verbosity(optuna.logging.INFO)


class ProphetOptimizer:
    """Optimize Prophet model hyperparameters using Optuna and backtesting"""
    
    def __init__(self, data_path='data.csv', n_trials=50, n_folds=3):
        """
        Initialize optimizer
        
        Parameters:
        -----------
        data_path : str
            Path to the data CSV file
        n_trials : int
            Number of Optuna trials to run
        n_folds : int
            Number of backtesting folds (cross-validation)
        """
        self.data_path = data_path
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.best_params = None
        self.best_score = None
        self.study = None
        
        # Load and prepare data
        self._load_data()
    
    def _load_data(self):
        """Load data and create TimeSeries objects"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create target series
        self.target_series = TimeSeries.from_dataframe(
            df,
            time_col='date',
            value_cols='product_price',
            freq='W-FRI'
        )
        
        # Create covariate series
        self.covariate_series = TimeSeries.from_dataframe(
            df,
            time_col='date',
            value_cols=['gdp_usa', 'gdp_china', 'gdp_eu', 'gdp_total',
                       'gdp_usa_yoy_change', 'gdp_china_yoy_change', 'gdp_eu_yoy_change'],
            freq='W-FRI'
        )
        
        # Use only training period for optimization (2010-2024)
        train_end = pd.Timestamp('2024-12-31')
        self.train_target = self.target_series.split_before(train_end)[0]
        self.train_covariates = self.covariate_series
        
        print(f"Data loaded: {len(self.train_target)} training weeks")
    
    def _backtest_model(self, params, verbose=False):
        """
        Perform backtesting with given hyperparameters
        
        Parameters:
        -----------
        params : dict
            Prophet hyperparameters to test
        verbose : bool
            Print progress information
            
        Returns:
        --------
        float : Average MAPE across all backtesting folds
        """
        from darts.utils.statistics import check_seasonality
        
        try:
            # Create Prophet model with given parameters
            model = Prophet(
                yearly_seasonality=params['yearly_seasonality'],
                weekly_seasonality=params['weekly_seasonality'],
                daily_seasonality=False,
                seasonality_mode=params['seasonality_mode'],
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
                changepoint_range=params.get('changepoint_range', 0.8),
            )
            
            # Backtesting configuration
            # Use expanding window: train on increasing amounts of data
            forecast_horizon = 52  # Predict 52 weeks (1 year) ahead
            stride = 26  # Move forward by 26 weeks between folds
            
            # Calculate start position for backtesting
            # Leave enough data for initial training (at least 2 years = 104 weeks)
            min_train_length = 104
            
            if len(self.train_target) < min_train_length + forecast_horizon:
                if verbose:
                    print(f"Not enough data for backtesting")
                return float('inf')
            
            # Perform backtesting
            # This trains model multiple times on historical data and validates on subsequent periods
            from darts import concatenate
            
            scores = []
            
            # Manual backtesting loop for better control
            for fold_idx in range(self.n_folds):
                # Calculate split point
                split_point = min_train_length + (fold_idx * stride)
                
                if split_point + forecast_horizon > len(self.train_target):
                    break
                
                # Split data
                train_fold = self.train_target[:split_point]
                test_fold = self.train_target[split_point:split_point + forecast_horizon]
                
                if len(test_fold) < forecast_horizon:
                    continue
                
                # Train model on this fold
                model.fit(train_fold, future_covariates=self.train_covariates)
                
                # Predict
                pred = model.predict(n=forecast_horizon, future_covariates=self.train_covariates)
                
                # Calculate MAPE for this fold
                fold_mape = mape(test_fold, pred)
                scores.append(fold_mape)
                
                if verbose:
                    print(f"  Fold {fold_idx + 1}: MAPE = {fold_mape:.2f}%")
            
            if not scores:
                return float('inf')
            
            # Return average MAPE across all folds
            avg_mape = np.mean(scores)
            
            if verbose:
                print(f"  Average MAPE: {avg_mape:.2f}%")
            
            return avg_mape
            
        except Exception as e:
            if verbose:
                print(f"Error in backtesting: {str(e)}")
            return float('inf')
    
    def _objective(self, trial):
        """
        Optuna objective function
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
            
        Returns:
        --------
        float : MAPE score (to minimize)
        """
        # Suggest hyperparameters
        params = {
            'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
            'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False]),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True),
            'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True),
            'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95),
        }
        
        print(f"\nTrial {trial.number + 1}/{self.n_trials}")
        print(f"Testing parameters: {params}")
        
        # Evaluate using backtesting
        mape_score = self._backtest_model(params, verbose=True)
        
        print(f"Trial {trial.number + 1} result: MAPE = {mape_score:.2f}%")
        
        return mape_score
    
    def optimize(self):
        """Run Optuna optimization"""
        print("="*80)
        print("STARTING HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        print("="*80)
        print(f"Number of trials: {self.n_trials}")
        print(f"Backtesting folds: {self.n_folds}")
        print(f"Optimization metric: MAPE (Mean Absolute Percentage Error)")
        print("="*80)
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='minimize',  # Minimize MAPE
            study_name='prophet_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        start_time = datetime.now()
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        end_time = datetime.now()
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETED!")
        print("="*80)
        print(f"Time taken: {end_time - start_time}")
        print(f"\nBest MAPE: {self.best_score:.2f}%")
        print("\nBest Parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print("="*80)
        
        return self.best_params, self.best_score
    
    def visualize_optimization(self, save_dir='optimization_results'):
        """Create visualization plots for optimization results"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nGenerating optimization visualizations...")
        
        # 1. Optimization History
        fig1 = plot_optimization_history(self.study)
        fig1.write_html(f'{save_dir}/optimization_history.html')
        print(f"  ✓ Optimization history saved")
        
        # 2. Parameter Importances
        try:
            fig2 = plot_param_importances(self.study)
            fig2.write_html(f'{save_dir}/param_importances.html')
            print(f"  ✓ Parameter importances saved")
        except:
            print(f"  ⚠ Could not generate parameter importances (need more trials)")
        
        # 3. Parallel Coordinate Plot
        try:
            fig3 = plot_parallel_coordinate(self.study)
            fig3.write_html(f'{save_dir}/parallel_coordinate.html')
            print(f"  ✓ Parallel coordinate plot saved")
        except:
            print(f"  ⚠ Could not generate parallel coordinate plot")
        
        # 4. Slice Plot
        try:
            fig4 = plot_slice(self.study)
            fig4.write_html(f'{save_dir}/slice_plot.html')
            print(f"  ✓ Slice plot saved")
        except:
            print(f"  ⚠ Could not generate slice plot")
        
        # 5. Custom matplotlib plot
        self._create_summary_plot(save_dir)
        
        print(f"\nAll visualizations saved to '{save_dir}/' directory")
    
    def _create_summary_plot(self, save_dir):
        """Create custom matplotlib summary plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get trial data
        trials_df = self.study.trials_dataframe()
        
        # Plot 1: MAPE over trials
        ax1 = axes[0, 0]
        ax1.plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6)
        ax1.axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.2f}%')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('MAPE (%)')
        ax1.set_title('Optimization Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter distributions (changepoint_prior_scale)
        ax2 = axes[0, 1]
        param_name = 'params_changepoint_prior_scale'
        if param_name in trials_df.columns:
            ax2.scatter(trials_df[param_name], trials_df['value'], alpha=0.6)
            best_val = self.best_params['changepoint_prior_scale']
            ax2.axvline(x=best_val, color='r', linestyle='--', label=f'Best: {best_val:.4f}')
            ax2.set_xlabel('changepoint_prior_scale')
            ax2.set_ylabel('MAPE (%)')
            ax2.set_title('Changepoint Prior Scale vs MAPE')
            ax2.set_xscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Seasonality mode comparison
        ax3 = axes[1, 0]
        param_name = 'params_seasonality_mode'
        if param_name in trials_df.columns:
            mode_mapes = trials_df.groupby(param_name)['value'].mean()
            mode_mapes.plot(kind='bar', ax=ax3, color=['#1f77b4', '#ff7f0e'])
            ax3.set_xlabel('Seasonality Mode')
            ax3.set_ylabel('Average MAPE (%)')
            ax3.set_title('Seasonality Mode Performance')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Best vs Default comparison (text summary)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
OPTIMIZATION SUMMARY

Total Trials: {len(trials_df)}
Best MAPE: {self.best_score:.2f}%

Best Configuration:
────────────────────
Seasonality Mode: {self.best_params['seasonality_mode']}
Yearly Seasonality: {self.best_params['yearly_seasonality']}
Weekly Seasonality: {self.best_params['weekly_seasonality']}
Changepoint Prior: {self.best_params['changepoint_prior_scale']:.4f}
Seasonality Prior: {self.best_params['seasonality_prior_scale']:.4f}
Holidays Prior: {self.best_params['holidays_prior_scale']:.4f}
Changepoint Range: {self.best_params['changepoint_range']:.2f}

Note: Lower MAPE is better
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/optimization_summary.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Summary plot saved")
    
    def save_best_params(self, filepath='best_params.json'):
        """Save best parameters to JSON file"""
        if self.best_params is None:
            print("No optimization results to save. Run optimize() first.")
            return
        
        results = {
            'best_params': self.best_params,
            'best_mape': self.best_score,
            'n_trials': self.n_trials,
            'n_folds': self.n_folds,
            'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Best parameters saved to '{filepath}'")
    
    def train_best_model(self):
        """Train final model with best parameters on full training data"""
        if self.best_params is None:
            print("No optimization results. Run optimize() first.")
            return None
        
        print("\n" + "="*80)
        print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print("="*80)
        
        # Create model with best parameters
        model = Prophet(
            yearly_seasonality=self.best_params['yearly_seasonality'],
            weekly_seasonality=self.best_params['weekly_seasonality'],
            daily_seasonality=False,
            seasonality_mode=self.best_params['seasonality_mode'],
            changepoint_prior_scale=self.best_params['changepoint_prior_scale'],
            seasonality_prior_scale=self.best_params['seasonality_prior_scale'],
            holidays_prior_scale=self.best_params['holidays_prior_scale'],
            changepoint_range=self.best_params['changepoint_range'],
        )
        
        print("Training on full dataset...")
        model.fit(self.train_target, future_covariates=self.train_covariates)
        
        print("✓ Model trained successfully!")
        
        return model


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("PROPHET HYPERPARAMETER OPTIMIZATION WITH OPTUNA + BACKTESTING")
    print("="*80)
    
    # Configuration
    N_TRIALS = 30  # Number of optimization trials (increase for better results)
    N_FOLDS = 3    # Number of backtesting folds
    
    print(f"\nConfiguration:")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Backtesting folds: {N_FOLDS}")
    print(f"  Evaluation metric: MAPE")
    print("="*80)
    
    # Create optimizer
    optimizer = ProphetOptimizer(
        data_path='data.csv',
        n_trials=N_TRIALS,
        n_folds=N_FOLDS
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize()
    
    # Visualize results
    optimizer.visualize_optimization()
    
    # Save best parameters
    optimizer.save_best_params('best_params.json')
    
    # Train final model with best parameters
    best_model = optimizer.train_best_model()
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. best_params.json - Best hyperparameters")
    print("  2. optimization_results/ - Visualization plots")
    print("     - optimization_history.html")
    print("     - param_importances.html")
    print("     - parallel_coordinate.html")
    print("     - slice_plot.html")
    print("     - optimization_summary.png")
    print("\nYou can now use these optimized parameters in your forecast_model.py")
    print("="*80)


if __name__ == "__main__":
    main()

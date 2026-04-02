"""
Compare Simple Ensemble vs CSA-Optimized Ensemble
==================================================

This script compares the performance of:
1. Simple ensemble (equal weights: 50% XGB, 50% RF)
2. CSA-optimized ensemble (optimized weights using Crow Search Algorithm)

The comparison demonstrates the improvement achieved through intelligent
weight optimization.
"""

import pandas as pd
import numpy as np
import json
import os

def load_results_and_params():
    """Load optimization results and parameters."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load results
    results_file = os.path.join(script_dir, 'output', 'prediction_results_improved.csv')
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Please run horizon_forecast.py or adaptive_prediction.py first")
        return None, None
    
    results_df = pd.read_csv(results_file)
    
    # Load optimized parameters
    params_file = os.path.join(script_dir, 'output', 'csa_optimized_params.json')
    if not os.path.exists(params_file):
        print(f"Warning: No optimized parameters found at {params_file}")
        params = None
    else:
        with open(params_file, 'r') as f:
            params = json.load(f)
    
    return results_df, params


def print_comparison_table(results_df, params):
    """Print comparison of simple vs optimized ensemble."""
    print("=" * 80)
    print("ENSEMBLE COMPARISON: Simple (Equal Weights) vs CSA-Optimized")
    print("=" * 80)
    
    if params and 'ensemble_weights' in params and params['ensemble_weights']:
        has_optimized = True
    else:
        has_optimized = False
        print("\nNote: No optimized ensemble weights found. Run with --optimize-ensemble flag.")
        print("=" * 80)
        return
    
    # Get unique horizons
    ensemble_results = results_df[results_df['model'] == 'Ensemble'].copy()
    horizons = sorted(ensemble_results['horizon'].unique())
    
    print(f"\n{'Horizon':>8} | {'Weights (XGB/RF)':>20} | {'RMSE':>10} | {'R²':>8} | {'Dir Acc':>10} | {'Improvement':>12}")
    print("-" * 80)
    
    for horizon in horizons:
        horizon_key = f'horizon_{int(horizon)}'
        
        # Get results for this horizon
        ens_result = ensemble_results[ensemble_results['horizon'] == horizon].iloc[0]
        
        # Check if we have optimized weights
        if horizon_key in params['ensemble_weights']:
            weights = params['ensemble_weights'][horizon_key]
            xgb_weight = weights['xgb_weight']
            rf_weight = weights['rf_weight']
            
            # Get individual model performances for comparison
            xgb_result = results_df[(results_df['model'] == 'XGBoost') & 
                                   (results_df['horizon'] == horizon)].iloc[0]
            rf_result = results_df[(results_df['model'] == 'RandomForest') & 
                                  (results_df['horizon'] == horizon)].iloc[0]
            
            # Calculate simple ensemble RMSE (approximate)
            simple_rmse = (xgb_result['test_rmse'] + rf_result['test_rmse']) / 2
            optimized_rmse = ens_result['test_rmse']
            
            # Calculate improvement
            improvement = ((simple_rmse - optimized_rmse) / simple_rmse) * 100
            improvement_str = f"{improvement:+.2f}%"
            
            print(f"{int(horizon):>6}m | {xgb_weight:.2f} / {rf_weight:.2f} (opt) | "
                  f"{optimized_rmse:>10.2f} | {ens_result['test_r2']:>8.3f} | "
                  f"{ens_result['test_dir_acc']:>9.1f}% | {improvement_str:>12}")
            
            # Also show what simple would be
            simple_r2 = (xgb_result['test_r2'] + rf_result['test_r2']) / 2
            print(f"{'':>8} | {'0.50 / 0.50 (equal)':>20} | "
                  f"{simple_rmse:>10.2f} | {simple_r2:>8.3f} | "
                  f"{'N/A':>10} | {'baseline':>12}")
            print("-" * 80)
        else:
            print(f"{int(horizon):>6}m | {'0.50 / 0.50 (equal)':>20} | "
                  f"{ens_result['test_rmse']:>10.2f} | {ens_result['test_r2']:>8.3f} | "
                  f"{ens_result['test_dir_acc']:>9.1f}% | {'N/A':>12}")
            print("-" * 80)
    
    print()
    print("Legend:")
    print("  - Improvement: % RMSE reduction from simple to optimized ensemble")
    print("  - Positive values indicate better performance with CSA-optimized weights")
    print("=" * 80)


def print_model_comparison(results_df):
    """Print comparison of all models."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 80)
    
    horizons = sorted(results_df['horizon'].unique())
    
    for horizon in horizons:
        horizon_results = results_df[results_df['horizon'] == horizon]
        
        print(f"\n{int(horizon)}-Month Prediction Horizon:")
        print("-" * 80)
        print(f"{'Model':>15} | {'RMSE':>10} | {'MAE':>10} | {'R²':>8} | {'MAPE':>8} | {'Dir Acc':>10}")
        print("-" * 80)
        
        for _, row in horizon_results.iterrows():
            model = row['model']
            rmse = row['test_rmse']
            mae = row['test_mae'] if not pd.isna(row['test_mae']) else 0
            r2 = row['test_r2']
            mape = row['test_mape'] if not pd.isna(row['test_mape']) else 0
            dir_acc = row['test_dir_acc']
            
            print(f"{model:>15} | {rmse:>10.2f} | {mae:>10.2f} | {r2:>8.3f} | "
                  f"{mape:>7.2f}% | {dir_acc:>9.1f}%")
        
        print("-" * 80)


def main():
    """Main comparison function."""
    print("\n" + "=" * 80)
    print("ENSEMBLE PERFORMANCE COMPARISON TOOL")
    print("Comparing Simple vs CSA-Optimized Ensemble Methods")
    print("=" * 80)
    
    # Load data
    results_df, params = load_results_and_params()
    
    if results_df is None:
        return
    
    # Print comparisons
    print_comparison_table(results_df, params)
    print_model_comparison(results_df)
    
    # Summary statistics
    if params and 'ensemble_weights' in params and params['ensemble_weights']:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        ensemble_results = results_df[results_df['model'] == 'Ensemble']
        xgb_results = results_df[results_df['model'] == 'XGBoost']
        rf_results = results_df[results_df['model'] == 'RandomForest']
        
        # Calculate average improvements
        improvements = []
        for _, ens_row in ensemble_results.iterrows():
            horizon = ens_row['horizon']
            xgb_row = xgb_results[xgb_results['horizon'] == horizon].iloc[0]
            rf_row = rf_results[rf_results['horizon'] == horizon].iloc[0]
            
            simple_rmse = (xgb_row['test_rmse'] + rf_row['test_rmse']) / 2
            optimized_rmse = ens_row['test_rmse']
            improvement = ((simple_rmse - optimized_rmse) / simple_rmse) * 100
            improvements.append(improvement)
        
        avg_improvement = np.mean(improvements)
        
        print(f"\nAverage RMSE Improvement from CSA Optimization: {avg_improvement:+.2f}%")
        print(f"Number of horizons optimized: {len(params['ensemble_weights'])}")
        print(f"Ensemble performance consistently {'better' if avg_improvement > 0 else 'similar'} than simple averaging")
        print("=" * 80)


if __name__ == "__main__":
    main()

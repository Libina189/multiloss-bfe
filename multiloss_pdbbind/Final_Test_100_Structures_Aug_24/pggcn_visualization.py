#!/usr/bin/env python3
"""
PGGCN Results Visualization Suite

This script provides comprehensive visualization functions for analyzing PGGCN training results.
It generates multiple types of plots to understand model performance, training dynamics, 
physics parameter effects, and prediction accuracy.

Usage:
1. Load your pickle file using load_and_explore_results()
2. Call generate_all_plots() to create all visualizations
3. Individual plot functions can be called separately for specific analyses

Plot Types Available:
- Training curves (loss, learning rate)
- Physics hyperparameter sweep analysis
- Prediction accuracy (scatter plots, residual analysis)
- Error distributions and statistics
- Comparative analysis across physics weights
- Data distribution analysis
- Model convergence analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import pickle
from math import sqrt

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_results(pickle_filename):
    """Load results from pickle file."""
    try:
        with open(pickle_filename, 'rb') as f:
            results_data = pickle.load(f)
        print(f"Loaded results from: {pickle_filename}")
        print(f"Dataset size: {results_data['experiment_info']['dataset_size']} structures")
        print(f"Number of physics weights tested: {len(results_data['all_results'])}")
        return results_data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def plot_training_curves(results_data, figsize=(15, 10)):
    """Plot training curves for all physics weights."""
    all_results = results_data['all_results']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Training Curves Analysis ({results_data["experiment_info"]["dataset_size"]} structures)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss for All Physics Weights
    ax = axes[0, 0]
    for result in all_results:
        if result['training_history']['total_losses']:
            epochs = range(1, len(result['training_history']['total_losses']) + 1)
            ax.plot(epochs, result['training_history']['total_losses'], 
                   label=f"λ={result['physics_weight']:.0e}", linewidth=2)
    ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Learning Rate Schedules
    ax = axes[0, 1]
    for result in all_results:
        if result['training_history']['learning_rates']:
            epochs = range(1, len(result['training_history']['learning_rates']) + 1)
            ax.plot(epochs, result['training_history']['learning_rates'], 
                   label=f"λ={result['physics_weight']:.0e}", linewidth=2)
    ax.set_title('Learning Rate Schedules', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rate')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Final Loss Comparison
    ax = axes[0, 2]
    physics_weights = [r['physics_weight'] for r in all_results]
    train_losses = [r['final_train_loss'] for r in all_results]
    test_losses = [r['test_loss'] for r in all_results]
    
    x_pos = range(len(physics_weights))
    width = 0.35
    ax.bar([x - width/2 for x in x_pos], train_losses, width, label='Train Loss', alpha=0.8)
    ax.bar([x + width/2 for x in x_pos], test_losses, width, label='Test Loss', alpha=0.8)
    
    ax.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Physics Weight')
    ax.set_ylabel('Loss')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{w:.0e}" for w in physics_weights], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence Analysis
    ax = axes[1, 0]
    for result in all_results:
        if result['training_history']['total_losses'] and len(result['training_history']['total_losses']) > 10:
            losses = result['training_history']['total_losses']
            # Calculate moving average
            window = min(10, len(losses) // 4)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            epochs = range(window, len(losses) + 1)
            ax.plot(epochs, moving_avg, label=f"λ={result['physics_weight']:.0e}", linewidth=2)
    ax.set_title('Training Convergence (Moving Average)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Smoothed Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 5: Training Efficiency
    ax = axes[1, 1]
    epochs_trained = [len(r['training_history']['total_losses']) for r in all_results if r['training_history']['total_losses']]
    final_losses = [r['final_train_loss'] for r in all_results if r['training_history']['total_losses']]
    physics_weights_eff = [r['physics_weight'] for r in all_results if r['training_history']['total_losses']]
    
    scatter = ax.scatter(epochs_trained, final_losses, 
                        c=range(len(epochs_trained)), s=100, alpha=0.7, cmap='viridis')
    for i, weight in enumerate(physics_weights_eff):
        ax.annotate(f"{weight:.0e}", (epochs_trained[i], final_losses[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_title('Training Efficiency', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs to Convergence')
    ax.set_ylabel('Final Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary Statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate summary statistics
    best_result = min(all_results, key=lambda x: x['test_loss'])
    runtime = results_data['experiment_info']['total_runtime_minutes']
    
    summary_text = f"""
    Dataset: {results_data['experiment_info']['dataset_size']} structures
    Runtime: {runtime:.1f} minutes
    
    Best Performance:
    Physics Weight: {best_result['physics_weight']:.0e}
    Test Loss: {best_result['test_loss']:.6f}
    Train Loss: {best_result['final_train_loss']:.6f}
    MAE: {best_result['mean_abs_diff']:.6f}
    
    Training Details:
    Epochs: {best_result['epochs_trained']}
    Batch Size: {best_result['config']['batch_size']}
    Max Padding: {best_result['config']['max_padding']}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax.set_title('Experiment Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_physics_hyperparameter_analysis(results_data, figsize=(15, 8)):
    """Analyze the effect of physics hyperparameter on model performance."""
    all_results = results_data['all_results']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Physics Hyperparameter Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    physics_weights = [r['physics_weight'] for r in all_results]
    train_losses = [r['final_train_loss'] for r in all_results]
    test_losses = [r['test_loss'] for r in all_results]
    maes = [r['mean_abs_diff'] for r in all_results]
    epochs_trained = [r['epochs_trained'] for r in all_results]
    
    # Sort by physics weight for better visualization
    sorted_indices = np.argsort(physics_weights)
    physics_weights = [physics_weights[i] for i in sorted_indices]
    train_losses = [train_losses[i] for i in sorted_indices]
    test_losses = [test_losses[i] for i in sorted_indices]
    maes = [maes[i] for i in sorted_indices]
    epochs_trained = [epochs_trained[i] for i in sorted_indices]
    
    # Plot 1: Loss vs Physics Weight
    ax = axes[0, 0]
    ax.semilogx(physics_weights, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=8)
    ax.semilogx(physics_weights, test_losses, 's-', label='Test Loss', linewidth=2, markersize=8)
    ax.set_title('Loss vs Physics Weight', fontsize=14, fontweight='bold')
    ax.set_xlabel('Physics Weight (λ)')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MAE vs Physics Weight
    ax = axes[0, 1]
    ax.semilogx(physics_weights, maes, 'o-', color='red', linewidth=2, markersize=8)
    ax.set_title('Mean Absolute Error vs Physics Weight', fontsize=14, fontweight='bold')
    ax.set_xlabel('Physics Weight (λ)')
    ax.set_ylabel('Mean Absolute Error')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Overfitting Analysis (Train-Test Gap)
    ax = axes[0, 2]
    overfitting_gap = np.array(test_losses) - np.array(train_losses)
    ax.semilogx(physics_weights, overfitting_gap, 'o-', color='purple', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Overfitting Analysis (Test - Train Loss)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Physics Weight (λ)')
    ax.set_ylabel('Loss Gap')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training Efficiency vs Physics Weight
    ax = axes[1, 0]
    ax.semilogx(physics_weights, epochs_trained, 'o-', color='green', linewidth=2, markersize=8)
    ax.set_title('Training Efficiency vs Physics Weight', fontsize=14, fontweight='bold')
    ax.set_xlabel('Physics Weight (λ)')
    ax.set_ylabel('Epochs to Convergence')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Performance Correlation Matrix
    ax = axes[1, 1]
    data_matrix = np.array([physics_weights, train_losses, test_losses, maes, epochs_trained]).T
    correlation_matrix = np.corrcoef(data_matrix.T)
    
    labels = ['Physics λ', 'Train Loss', 'Test Loss', 'MAE', 'Epochs']
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    
    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                   ha='center', va='center', fontweight='bold')
    
    ax.set_title('Performance Correlation Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Plot 6: Optimal Physics Weight Analysis
    ax = axes[1, 2]
    
    # Normalize metrics for comparison (0-1 scale)
    norm_train = (np.array(train_losses) - min(train_losses)) / (max(train_losses) - min(train_losses))
    norm_test = (np.array(test_losses) - min(test_losses)) / (max(test_losses) - min(test_losses))
    norm_mae = (np.array(maes) - min(maes)) / (max(maes) - min(maes))
    
    # Combined score (lower is better)
    combined_score = norm_train + norm_test + norm_mae
    
    ax.semilogx(physics_weights, combined_score, 'o-', color='gold', linewidth=3, markersize=10)
    best_idx = np.argmin(combined_score)
    ax.semilogx(physics_weights[best_idx], combined_score[best_idx], 
               'r*', markersize=20, label=f'Optimal: {physics_weights[best_idx]:.0e}')
    
    ax.set_title('Combined Performance Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Physics Weight (λ)')
    ax.set_ylabel('Normalized Combined Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_prediction_analysis(results_data, figsize=(20, 12)):
    """Comprehensive prediction accuracy analysis."""
    all_results = results_data['all_results']
    n_results = len(all_results)
    
    # Create subplots: 3 rows x n_results columns
    fig, axes = plt.subplots(3, n_results, figsize=figsize)
    if n_results == 1:
        axes = axes.reshape(3, 1)
    
    fig.suptitle('Prediction Analysis for All Physics Weights', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(all_results):
        physics_weight = result['physics_weight']
        y_true_test = result['y_true_test']
        y_pred_test = result['y_pred_test']
        y_true_train = result['y_true_train']
        y_pred_train = result['y_pred_train']
        
        # Calculate metrics
        r2_test = r2_score(y_true_test, y_pred_test)
        r2_train = r2_score(y_true_train, y_pred_train)
        rmse_test = sqrt(mean_squared_error(y_true_test, y_pred_test))
        rmse_train = sqrt(mean_squared_error(y_true_train, y_pred_train))
        mae_test = mean_absolute_error(y_true_test, y_pred_test)
        mae_train = mean_absolute_error(y_true_train, y_pred_train)
        
        # Plot 1: Test Set Predictions (Row 0)
        ax = axes[0, i]
        ax.scatter(y_true_test, y_pred_test, alpha=0.7, s=60)
        
        # Perfect prediction line
        min_val = min(min(y_true_test), min(y_pred_test))
        max_val = max(max(y_true_test), max(y_pred_test))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_test, y_pred_test)
        line_x = np.linspace(min_val, max_val, 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
        
        ax.set_xlabel('True ΔG')
        ax.set_ylabel('Predicted ΔG')
        ax.set_title(f'Test Predictions (λ={physics_weight:.0e})\nR²={r2_test:.3f}, RMSE={rmse_test:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Training Set Predictions (Row 1)
        ax = axes[1, i]
        ax.scatter(y_true_train, y_pred_train, alpha=0.7, s=60, color='orange')
        
        min_val = min(min(y_true_train), min(y_pred_train))
        max_val = max(max(y_true_train), max(y_pred_train))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_train, y_pred_train)
        line_x = np.linspace(min_val, max_val, 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
        
        ax.set_xlabel('True ΔG')
        ax.set_ylabel('Predicted ΔG')
        ax.set_title(f'Train Predictions (λ={physics_weight:.0e})\nR²={r2_train:.3f}, RMSE={rmse_train:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Residual Analysis (Row 2)
        ax = axes[2, i]
        residuals_test = y_true_test - y_pred_test
        residuals_train = y_true_train - y_pred_train
        
        ax.scatter(y_pred_test, residuals_test, alpha=0.7, s=60, label='Test', color='blue')
        ax.scatter(y_pred_train, residuals_train, alpha=0.7, s=60, label='Train', color='orange')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Predicted ΔG')
        ax.set_ylabel('Residuals (True - Predicted)')
        ax.set_title(f'Residual Analysis (λ={physics_weight:.0e})\nTest MAE={mae_test:.3f}, Train MAE={mae_train:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_error_distributions(results_data, figsize=(15, 10)):
    """Analyze error distributions and statistical properties."""
    all_results = results_data['all_results']
    n_results = len(all_results)
    
    fig, axes = plt.subplots(2, n_results, figsize=figsize)
    if n_results == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(all_results):
        physics_weight = result['physics_weight']
        y_true_test = result['y_true_test']
        y_pred_test = result['y_pred_test']
        
        errors = y_true_test - y_pred_test
        abs_errors = np.abs(errors)
        
        # Plot 1: Error Histogram
        ax = axes[0, i]
        ax.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
        ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.3f}')
        
        ax.set_xlabel('Prediction Error (True - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution (λ={physics_weight:.0e})\nStd: {np.std(errors):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Q-Q Plot for Normality Check
        ax = axes[1, i]
        stats.probplot(errors, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot (λ={physics_weight:.0e})\nNormality Test')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_data_distribution_analysis(results_data, figsize=(15, 8)):
    """Analyze the distribution of true values and predictions."""
    # Get the best result for detailed analysis
    best_result = min(results_data['all_results'], key=lambda x: x['test_loss'])
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Data Distribution Analysis (Best Model: λ={best_result["physics_weight"]:.0e})', 
                 fontsize=16, fontweight='bold')
    
    y_true_test = best_result['y_true_test']
    y_pred_test = best_result['y_pred_test']
    y_true_train = best_result['y_true_train']
    y_pred_train = best_result['y_pred_train']
    
    # Plot 1: True Values Distribution
    ax = axes[0, 0]
    ax.hist(y_true_train, bins=20, alpha=0.7, label='Train', color='orange')
    ax.hist(y_true_test, bins=20, alpha=0.7, label='Test', color='blue')
    ax.set_xlabel('True ΔG Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of True Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Predicted Values Distribution
    ax = axes[0, 1]
    ax.hist(y_pred_train, bins=20, alpha=0.7, label='Train', color='orange')
    ax.hist(y_pred_test, bins=20, alpha=0.7, label='Test', color='blue')
    ax.set_xlabel('Predicted ΔG Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predicted Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Train vs Test Comparison
    ax = axes[0, 2]
    ax.scatter(y_true_train, y_pred_train, alpha=0.6, label='Train', s=30, color='orange')
    ax.scatter(y_true_test, y_pred_test, alpha=0.8, label='Test', s=50, color='blue')
    
    # Perfect prediction line
    all_true = np.concatenate([y_true_train, y_true_test])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    min_val, max_val = min(min(all_true), min(all_pred)), max(max(all_true), max(all_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('True ΔG')
    ax.set_ylabel('Predicted ΔG')
    ax.set_title('Train vs Test Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error vs True Values
    ax = axes[1, 0]
    train_errors = y_true_train - y_pred_train
    test_errors = y_true_test - y_pred_test
    
    ax.scatter(y_true_train, train_errors, alpha=0.6, label='Train', s=30, color='orange')
    ax.scatter(y_true_test, test_errors, alpha=0.8, label='Test', s=50, color='blue')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('True ΔG')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Error vs True Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Absolute Error vs True Values
    ax = axes[1, 1]
    train_abs_errors = np.abs(train_errors)
    test_abs_errors = np.abs(test_errors)
    
    ax.scatter(y_true_train, train_abs_errors, alpha=0.6, label='Train', s=30, color='orange')
    ax.scatter(y_true_test, test_abs_errors, alpha=0.8, label='Test', s=50, color='blue')
    
    ax.set_xlabel('True ΔG')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Error vs True Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Statistics Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate comprehensive statistics
    r2_test = r2_score(y_true_test, y_pred_test)
    r2_train = r2_score(y_true_train, y_pred_train)
    rmse_test = sqrt(mean_squared_error(y_true_test, y_pred_test))
    rmse_train = sqrt(mean_squared_error(y_true_train, y_pred_train))
    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    mae_train = mean_absolute_error(y_true_train, y_pred_train)
    
    stats_text = f"""
    Performance Statistics:
    
    Test Set:
    R²: {r2_test:.4f}
    RMSE: {rmse_test:.4f}
    MAE: {mae_test:.4f}
    
    Train Set:
    R²: {r2_train:.4f}
    RMSE: {rmse_train:.4f}
    MAE: {mae_train:.4f}
    
    Data Statistics:
    Train samples: {len(y_true_train)}
    Test samples: {len(y_true_test)}
    ΔG range: [{min(all_true):.2f}, {max(all_true):.2f}]
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def generate_all_plots(results_data, save_plots=True, output_dir="/home/exouser/multiloss-bfe/multiloss_pdbbind/Final_Test_100_Structures_Aug_24/claude"):
    """Generate all visualization plots and optionally save them."""
    plots = {}
    
    print("Generating comprehensive visualization suite...")
    
    # 1. Training Curves Analysis
    print("  1/5 Creating training curves analysis...")
    plots['training_curves'] = plot_training_curves(results_data)
    
    # 2. Physics Hyperparameter Analysis
    print("  2/5 Creating physics hyperparameter analysis...")
    plots['physics_analysis'] = plot_physics_hyperparameter_analysis(results_data)
    
    # 3. Prediction Analysis
    print("  3/5 Creating prediction analysis...")
    plots['prediction_analysis'] = plot_prediction_analysis(results_data)
    
    # 4. Error Distribution Analysis
    print("  4/5 Creating error distribution analysis...")
    plots['error_distributions'] = plot_error_distributions(results_data)
    
    # 5. Data Distribution Analysis
    print("  5/5 Creating data distribution analysis...")
    plots['data_distributions'] = plot_data_distribution_analysis(results_data)
    
    if save_plots:
        dataset_size = results_data['experiment_info']['dataset_size']
        timestamp = results_data['experiment_info']['timestamp']
        
        for plot_name, fig in plots.items():
            filename = f"{output_dir}PGGCN_{plot_name}_{dataset_size}structures_{timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
    
    print("All plots generated successfully!")
    return plots

# Example usage function
def example_usage():
    """Example of how to use the visualization functions."""
    print("PGGCN Visualization Suite - Example Usage")
    print("=" * 50)
    print()
    print("1. Load your results:")
    print("   results_data = load_results('/home/exouser/multiloss-bfe/multiloss_pdbbind/Final_Test_100_Structures_Aug_24/PGGCN_results_100structures_20250825_034719.pkl')")
    print()
    print("2. Generate all plots at once:")
    print("   plots = generate_all_plots(results_data, save_plots=True)")
    print()
    print("3. Or generate individual plots:")
    print("   fig1 = plot_training_curves(results_data)")
    print("   fig2 = plot_physics_hyperparameter_analysis(results_data)")
    print("   fig3 = plot_prediction_analysis(results_data)")
    print("   fig4 = plot_error_distributions(results_data)")
    print("   fig5 = plot_data_distribution_analysis(results_data)")
    print()
    print("4. Show plots:")
    print("   plt.show()")

def create_comprehensive_report(results_data, save_report=True, output_dir="/home/exouser/multiloss-bfe/multiloss_pdbbind/Final_Test_100_Structures_Aug_24/claude"):
    """Create a comprehensive analysis report with all visualizations."""
    dataset_size = results_data['experiment_info']['dataset_size']
    timestamp = results_data['experiment_info']['timestamp']
    
    # Generate all plots
    plots = generate_all_plots(results_data, save_plots=save_report, output_dir=output_dir)
    
    # Create summary report
    best_result = min(results_data['all_results'], key=lambda x: x['test_loss'])
    
    report = f"""
PGGCN Training Results - Comprehensive Report
============================================

Experiment Details:
- Dataset Size: {dataset_size} structures  
- Training Date: {timestamp}
- Total Runtime: {results_data['experiment_info']['total_runtime_minutes']:.2f} minutes
- Physics Weights Tested: {len(results_data['all_results'])}

Best Model Performance:
- Optimal Physics Weight (λ): {best_result['physics_weight']:.2e}
- Test Loss: {best_result['test_loss']:.6f}
- Training Loss: {best_result['final_train_loss']:.6f}
- Mean Absolute Error: {best_result['mean_abs_diff']:.6f}
- Training Epochs: {best_result['epochs_trained']}

Model Configuration:
- Batch Size: {best_result['config']['batch_size']}
- Max Padding: {best_result['config']['max_padding']} atoms
- Memory Limit: {best_result['config']['memory_limit_gb']} GB

Performance Analysis:
1. Training Curves: Shows convergence behavior and learning dynamics
2. Physics Analysis: Demonstrates optimal physics weight selection
3. Prediction Accuracy: Evaluates model predictions vs true values
4. Error Analysis: Analyzes prediction error distributions
5. Data Distribution: Examines train/test data characteristics

Generated Visualizations:
- training_curves: Loss curves and convergence analysis
- physics_analysis: Physics hyperparameter optimization
- prediction_analysis: Scatter plots and residual analysis
- error_distributions: Error histograms and normality tests
- data_distributions: Data and prediction distributions

Recommendations:
- Best physics weight: {best_result['physics_weight']:.2e}
- Model shows {'good' if best_result['test_loss'] < 1.0 else 'moderate'} generalization
- Training {'converged well' if best_result['epochs_trained'] < 80 else 'may need more epochs'}
"""
    
    if save_report:
        report_filename = f"{output_dir}PGGCN_analysis_report_{dataset_size}structures_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"Comprehensive report saved: {report_filename}")
    
    print(report)
    return report, plots

# Interactive analysis functions
def compare_physics_weights(results_data, metrics=['test_loss', 'mean_abs_diff'], figsize=(12, 6)):
    """Create focused comparison of physics weights for specific metrics."""
    all_results = results_data['all_results']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    fig.suptitle('Physics Weight Comparison', fontsize=16, fontweight='bold')
    
    physics_weights = [r['physics_weight'] for r in all_results]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [r[metric] for r in all_results]
        
        # Sort by physics weight
        sorted_indices = np.argsort(physics_weights)
        sorted_weights = [physics_weights[j] for j in sorted_indices]
        sorted_values = [values[j] for j in sorted_indices]
        
        # Plot
        ax.semilogx(sorted_weights, sorted_values, 'o-', linewidth=2, markersize=8)
        
        # Highlight best value
        best_idx = np.argmin(sorted_values)
        ax.semilogx(sorted_weights[best_idx], sorted_values[best_idx], 
                   'r*', markersize=15, label=f'Best: {sorted_weights[best_idx]:.0e}')
        
        ax.set_xlabel('Physics Weight (λ)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Physics Weight')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_best_model_detailed(results_data, figsize=(20, 15)):
    """Detailed analysis of the best performing model."""
    best_result = min(results_data['all_results'], key=lambda x: x['test_loss'])
    
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.suptitle(f'Best Model Detailed Analysis (λ={best_result["physics_weight"]:.0e})', 
                 fontsize=18, fontweight='bold')
    
    y_true_test = best_result['y_true_test']
    y_pred_test = best_result['y_pred_test']
    y_true_train = best_result['y_true_train']
    y_pred_train = best_result['y_pred_train']
    
    # Calculate comprehensive metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2_test = r2_score(y_true_test, y_pred_test)
    r2_train = r2_score(y_true_train, y_pred_train)
    rmse_test = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    rmse_train = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    
    # Row 1: Prediction Analysis
    # Test predictions
    ax = axes[0, 0]
    ax.scatter(y_true_test, y_pred_test, alpha=0.7, s=60, color='blue')
    min_val = min(min(y_true_test), min(y_pred_test))
    max_val = max(max(y_true_test), max(y_pred_test))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('True ΔG')
    ax.set_ylabel('Predicted ΔG')
    ax.set_title(f'Test Set Predictions\nR²={r2_test:.3f}, RMSE={rmse_test:.3f}')
    ax.grid(True, alpha=0.3)
    
    # Training predictions
    ax = axes[0, 1]
    ax.scatter(y_true_train, y_pred_train, alpha=0.7, s=60, color='orange')
    min_val = min(min(y_true_train), min(y_pred_train))
    max_val = max(max(y_true_train), max(y_pred_train))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('True ΔG')
    ax.set_ylabel('Predicted ΔG')
    ax.set_title(f'Training Set Predictions\nR²={r2_train:.3f}, RMSE={rmse_train:.3f}')
    ax.grid(True, alpha=0.3)
    
    # Combined predictions
    ax = axes[0, 2]
    ax.scatter(y_true_train, y_pred_train, alpha=0.5, s=30, color='orange', label='Train')
    ax.scatter(y_true_test, y_pred_test, alpha=0.8, s=60, color='blue', label='Test')
    all_true = np.concatenate([y_true_train, y_true_test])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    min_val, max_val = min(min(all_true), min(all_pred)), max(max(all_true), max(all_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True ΔG')
    ax.set_ylabel('Predicted ΔG')
    ax.set_title('Combined Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training curve
    ax = axes[0, 3]
    if best_result['training_history']['total_losses']:
        epochs = range(1, len(best_result['training_history']['total_losses']) + 1)
        ax.plot(epochs, best_result['training_history']['total_losses'], 'b-', linewidth=2)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Training Loss')
        ax.set_title(f'Training Curve\nFinal Loss: {best_result["final_train_loss"]:.6f}')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Row 2: Error Analysis
    test_errors = y_true_test - y_pred_test
    train_errors = y_true_train - y_pred_train
    
    # Error distributions
    ax = axes[1, 0]
    ax.hist(test_errors, bins=15, alpha=0.7, color='blue', label='Test')
    ax.hist(train_errors, bins=15, alpha=0.7, color='orange', label='Train')
    ax.axvline(np.mean(test_errors), color='blue', linestyle='--', linewidth=2)
    ax.axvline(np.mean(train_errors), color='orange', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual plots
    ax = axes[1, 1]
    ax.scatter(y_pred_test, test_errors, alpha=0.7, s=60, color='blue', label='Test')
    ax.scatter(y_pred_train, train_errors, alpha=0.5, s=30, color='orange', label='Train')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted ΔG')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Absolute errors vs true values
    ax = axes[1, 2]
    ax.scatter(y_true_test, np.abs(test_errors), alpha=0.7, s=60, color='blue', label='Test')
    ax.scatter(y_true_train, np.abs(train_errors), alpha=0.5, s=30, color='orange', label='Train')
    ax.set_xlabel('True ΔG')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Error vs True Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot for error normality
    ax = axes[1, 3]
    stats.probplot(test_errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Test Errors)')
    ax.grid(True, alpha=0.3)
    
    # Row 3: Advanced Analysis
    # Learning rate curve
    ax = axes[2, 0]
    if best_result['training_history']['learning_rates']:
        epochs = range(1, len(best_result['training_history']['learning_rates']) + 1)
        ax.plot(epochs, best_result['training_history']['learning_rates'], 'g-', linewidth=2)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Error percentiles
    ax = axes[2, 1]
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    test_percentiles = [np.percentile(np.abs(test_errors), p) for p in percentiles]
    train_percentiles = [np.percentile(np.abs(train_errors), p) for p in percentiles]
    
    ax.plot(percentiles, test_percentiles, 'o-', color='blue', label='Test', linewidth=2, markersize=8)
    ax.plot(percentiles, train_percentiles, 's-', color='orange', label='Train', linewidth=2, markersize=8)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Percentiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Data range analysis
    ax = axes[2, 2]
    # Bin the true values and calculate metrics for each bin
    bins = np.linspace(min(all_true), max(all_true), 5)
    bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    test_bin_errors = []
    train_bin_errors = []
    
    for i in range(len(bins)-1):
        test_mask = (y_true_test >= bins[i]) & (y_true_test < bins[i+1])
        train_mask = (y_true_train >= bins[i]) & (y_true_train < bins[i+1])
        
        if np.sum(test_mask) > 0:
            test_bin_errors.append(np.mean(np.abs(test_errors[test_mask])))
        else:
            test_bin_errors.append(0)
            
        if np.sum(train_mask) > 0:
            train_bin_errors.append(np.mean(np.abs(train_errors[train_mask])))
        else:
            train_bin_errors.append(0)
    
    ax.plot(bin_centers, test_bin_errors, 'o-', color='blue', label='Test', linewidth=2, markersize=8)
    ax.plot(bin_centers, train_bin_errors, 's-', color='orange', label='Train', linewidth=2, markersize=8)
    ax.set_xlabel('True ΔG Range')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error by ΔG Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[2, 3]
    ax.axis('off')
    
    summary_stats = f"""
    Best Model Summary:
    Physics Weight: {best_result['physics_weight']:.2e}
    
    Test Performance:
    R²: {r2_test:.4f}
    RMSE: {rmse_test:.4f}
    MAE: {mean_absolute_error(y_true_test, y_pred_test):.4f}
    
    Train Performance:
    R²: {r2_train:.4f}
    RMSE: {rmse_train:.4f}
    MAE: {mean_absolute_error(y_true_train, y_pred_train):.4f}
    
    Training Info:
    Epochs: {best_result['epochs_trained']}
    Final Loss: {best_result['final_train_loss']:.6f}
    Test Loss: {best_result['test_loss']:.6f}
    
    Data Info:
    Train: {len(y_true_train)} samples
    Test: {len(y_true_test)} samples
    ΔG Range: [{min(all_true):.2f}, {max(all_true):.2f}]
    """
    
    ax.text(0.05, 0.95, summary_stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    example_usage()
    print()
    print("Available functions:")
    print("- load_results(filename): Load pickle file")
    print("- generate_all_plots(results_data): Create all visualizations")
    print("- plot_training_curves(results_data): Training analysis")  
    print("- plot_physics_hyperparameter_analysis(results_data): Physics weight optimization")
    print("- plot_prediction_analysis(results_data): Prediction accuracy analysis")
    print("- plot_error_distributions(results_data): Error distribution analysis")
    print("- plot_data_distribution_analysis(results_data): Data distribution analysis")
    print("- compare_physics_weights(results_data): Focused physics weight comparison")
    print("- plot_best_model_detailed(results_data): Detailed best model analysis")
    print("- create_comprehensive_report(results_data): Full analysis report")
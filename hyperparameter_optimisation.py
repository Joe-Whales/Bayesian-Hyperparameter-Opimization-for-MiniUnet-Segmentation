# hyperparameter_optimization.py
import numpy as np
import time
import itertools
from tqdm import tqdm
import torch
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Callable, Union

def run_trial(
    hyperparameters: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    train_model_fn: Callable,
    get_model_fn: Callable,
    criterion_fn: Callable,
    device: torch.device
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run a single trial with given hyperparameters and return metrics.
    
    Args:
        hyperparameters: Dictionary of hyperparameters
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        train_model_fn: Function to train the model
        get_model_fn: Function to get model
        criterion_fn: Function to get loss criterion
        device: Device to train on
        
    Returns:
        test_metrics: Metrics on test set
        history: Training history
    """
    # Extract hyperparameters
    model_config = {
        'base_filters': int(hyperparameters['base_filters']),
        'depth': int(hyperparameters['depth']),
        'bilinear': hyperparameters['bilinear']
    }
    
    train_config = {
        'learning_rate': hyperparameters['learning_rate'],
        'weight_decay': hyperparameters['weight_decay'],
        'batch_size': int(hyperparameters['batch_size']),
        'num_epochs': 50,  # Can be fixed
        'patience': 10     # Can be fixed
    }
    
    # Initialize model
    model = get_model_fn(model_config, device=device)
    
    # Initialize criterion
    criterion = criterion_fn()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    # Train model
    model, history = train_model_fn(
        model, train_loader, val_loader, optimizer, criterion,
        scheduler=scheduler, 
        num_epochs=train_config['num_epochs'],
        patience=train_config['patience'],
        device=device,
        verbose=False  # Suppress output
    )
    
    # Evaluate on test set (if a test_loader is provided)
    from train_utils import evaluate
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # Clean up to prevent memory leaks
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    
    return test_metrics, history


def perform_grid_search(
    param_space: Dict[str, List[Any]],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    train_model_fn: Callable,
    get_model_fn: Callable,
    criterion_fn: Callable,
    device: torch.device,
    metric: str = 'dice_score'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform grid search over hyperparameter space.
    
    Args:
        param_space: Dictionary of hyperparameter lists to search
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        train_model_fn: Function to train the model
        get_model_fn: Function to get model
        criterion_fn: Function to get loss criterion
        device: Device to train on
        metric: Metric to optimize
        
    Returns:
        results_df: DataFrame with all results
        best_result: Dictionary with best hyperparameters and metrics
    """
    # Get all combinations of hyperparameters
    param_keys = list(param_space.keys())
    param_values = list(param_space.values())
    param_combinations = list(itertools.product(*param_values))
    
    # Initialize results list
    results = []
    
    # Run trials
    print(f"Starting grid search with {len(param_combinations)} combinations")
    start_time = time.time()
    
    for i, values in enumerate(tqdm(param_combinations, desc="Grid Search Progress")):
        hyperparameters = dict(zip(param_keys, values))
        
        # Estimate time remaining
        if i > 0:
            elapsed = time.time() - start_time
            avg_time_per_trial = elapsed / i
            remaining_trials = len(param_combinations) - i
            estimated_time = avg_time_per_trial * remaining_trials
            hrs, remainder = divmod(estimated_time, 3600)
            mins, secs = divmod(remainder, 60)
            time_str = f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
            tqdm.write(f"Trial {i+1}/{len(param_combinations)}, Est. time remaining: {time_str}")
        
        # Run trial
        test_metrics, history = run_trial(
            hyperparameters,
            train_loader, val_loader, test_loader,
            train_model_fn, get_model_fn, criterion_fn, device
        )
        
        # Store results
        result = {**hyperparameters, **test_metrics, 'best_epoch': len(history['train_loss'])}
        results.append(result)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Get best result based on specified metric
    best_idx = results_df[metric].argmax() if 'score' in metric else results_df[metric].argmin()
    best_result = results_df.iloc[best_idx].to_dict()
    
    print(f"\nGrid search completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best {metric}: {best_result[metric]:.4f}")
    
    return results_df, best_result


def perform_bayesian_optimization(
    param_space: Dict[str, Union[Tuple[float, float], List[Any]]],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    train_model_fn: Callable,
    get_model_fn: Callable,
    criterion_fn: Callable,
    device: torch.device,
    n_trials: int = 35,
    metric: str = 'dice_score',
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform Bayesian hyperparameter optimization.
    
    Args:
        param_space: Dictionary of parameter ranges (tuple) or lists
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        train_model_fn: Function to train the model
        get_model_fn: Function to get model
        criterion_fn: Function to get loss criterion
        device: Device to train on
        n_trials: Number of trials to run
        metric: Metric to optimize
        seed: Random seed for reproducibility
        
    Returns:
        results_df: DataFrame with all results
        best_result: Dictionary with best hyperparameters and metrics
    """
    # Store results
    results = []
    
    # Define objective function for Optuna
    def objective(trial):
        # Sample hyperparameters
        hyperparameters = {}
        for key, value in param_space.items():
            if isinstance(value, list):
                # Categorical parameter
                hyperparameters[key] = trial.suggest_categorical(key, value)
            elif isinstance(value, tuple) and len(value) == 2:
                # Check if parameter is for integer values
                if key in ['base_filters', 'depth', 'batch_size']:
                    hyperparameters[key] = trial.suggest_int(key, int(value[0]), int(value[1]))
                # Check if parameter should use log scale
                elif key in ['learning_rate', 'weight_decay']:
                    hyperparameters[key] = trial.suggest_float(key, value[0], value[1], log=True)
                else:
                    hyperparameters[key] = trial.suggest_float(key, value[0], value[1])
            else:
                raise ValueError(f"Invalid parameter space format for {key}: {value}")
        
        # Run trial
        test_metrics, history = run_trial(
            hyperparameters,
            train_loader, val_loader, test_loader,
            train_model_fn, get_model_fn, criterion_fn, device
        )
        
        # Store results
        result = {**hyperparameters, **test_metrics, 'best_epoch': len(history['train_loss'])}
        results.append(result)
        
        # Return metric to optimize
        return test_metrics[metric] if 'score' in metric else -test_metrics[metric]
    
    # Create study
    direction = "maximize" if "score" in metric else "minimize"
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    
    # Run optimization
    print(f"Starting Bayesian optimization with {n_trials} trials")
    start_time = time.time()
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Get best result
    best_result = results_df.iloc[study.best_trial.number].to_dict() if results else None
    
    print(f"\nBayesian optimization completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best {metric}: {best_result[metric]:.4f}")
    
    return results_df, best_result


def plot_optimization_comparison(
    grid_results: pd.DataFrame,
    bayesian_results: pd.DataFrame,
    metric: str = 'dice_score',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot comparison of grid search and Bayesian optimization results.
    
    Args:
        grid_results: DataFrame with grid search results
        bayesian_results: DataFrame with Bayesian optimization results
        metric: Metric to compare
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Track the best score seen so far
    grid_best = grid_results[metric].copy()
    if 'score' not in metric:  # For metrics where lower is better
        grid_best = -grid_best
    grid_cummax = grid_best.cummax() if 'score' in metric else grid_best.cummin()
    if 'score' not in metric:
        grid_cummax = -grid_cummax
    
    bayesian_best = bayesian_results[metric].copy()
    if 'score' not in metric:  # For metrics where lower is better
        bayesian_best = -bayesian_best
    bayesian_cummax = bayesian_best.cummax() if 'score' in metric else bayesian_best.cummin()
    if 'score' not in metric:
        bayesian_cummax = -bayesian_cummax
    
    # Plot results
    plt.plot(grid_cummax.index, grid_cummax.values, 'b-', label='Grid Search')
    plt.plot(bayesian_cummax.index, bayesian_cummax.values, 'r-', label='Bayesian Optimization')
    
    # Add markers for each evaluation
    plt.scatter(grid_results.index, grid_results[metric], c='blue', alpha=0.5, s=20)
    plt.scatter(bayesian_results.index, bayesian_results[metric], c='red', alpha=0.5, s=20)
    
    # Add best found values
    grid_best_val = grid_cummax.max() if 'score' in metric else grid_cummax.min()
    bayesian_best_val = bayesian_cummax.max() if 'score' in metric else bayesian_cummax.min()
    
    title = f'Optimization Comparison\nBest Grid: {grid_best_val:.4f}, Best Bayesian: {bayesian_best_val:.4f}'
    plt.title(title)
    plt.xlabel('Trial')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add trial counts in the legend
    plt.figtext(0.01, 0.01, f'Grid trials: {len(grid_results)}, Bayesian trials: {len(bayesian_results)}', 
                ha='left', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_hyperparameter_importance(
    bayesian_results: pd.DataFrame, 
    optimized_metric: str = 'dice_score',
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot hyperparameter importance based on correlation with performance.
    
    Args:
        bayesian_results: DataFrame with Bayesian optimization results
        optimized_metric: Metric that was optimized
        figsize: Figure size
    """
    # Calculate correlation of each hyperparameter with the metric
    numeric_columns = bayesian_results.select_dtypes(include=[np.number]).columns
    hyperparams = [col for col in numeric_columns if col != optimized_metric and 'loss' not in col and 'score' not in col]
    
    correlations = []
    for param in hyperparams:
        correlation = bayesian_results[param].corr(bayesian_results[optimized_metric])
        correlations.append((param, abs(correlation), correlation))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Plot
    plt.figure(figsize=figsize)
    params = [c[0] for c in correlations]
    abs_corrs = [c[1] for c in correlations]
    corr_signs = [c[2] > 0 for c in correlations]
    
    colors = ['green' if sign else 'red' for sign in corr_signs]
    
    plt.barh(params, abs_corrs, color=colors)
    plt.xlabel('Absolute Correlation with Performance')
    plt.ylabel('Hyperparameter')
    plt.title('Hyperparameter Importance')
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Positive Correlation'),
        Patch(facecolor='red', label='Negative Correlation')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()
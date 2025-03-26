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
        optimizer, mode='min', factor=0.5, patience=5
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
    train_dataset,
    val_dataset,
    test_dataset,
    train_model_fn: Callable,
    get_model_fn: Callable,
    criterion_fn: Callable,
    device: torch.device,
    metric: str = 'iou_score',
    num_workers: int = 4
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform grid search with batch size as the outermost loop.
    """
    # Extract batch size options and remove from param_space
    batch_sizes = param_space.pop('batch_size')
    
    # Get all combinations of remaining hyperparameters
    param_keys = list(param_space.keys())
    param_values = list(param_space.values())
    param_combinations = list(itertools.product(*param_values))
    
    # Initialize results list
    results = []
    
    # Run trials
    total_trials = len(batch_sizes) * len(param_combinations)
    print(f"Starting grid search with {total_trials} combinations")
    start_time = time.time()
    trial_counter = 0
    
    # Outermost loop over batch sizes
    for batch_size in batch_sizes:
        # Create data loaders only once per batch size
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Inner loop over all other hyperparameter combinations
        for values in tqdm(param_combinations, 
                          desc=f"Grid Search (batch_size={batch_size})", 
                          leave=False):
            trial_counter += 1
            
            # Build full hyperparameter set
            hyperparameters = dict(zip(param_keys, values))
            hyperparameters['batch_size'] = batch_size
            
            # Run trial
            test_metrics, history = run_trial(
                hyperparameters,
                train_loader, val_loader, test_loader,
                train_model_fn, get_model_fn, criterion_fn, device
            )
            
            # Store results
            result = {**hyperparameters, **test_metrics, 'best_epoch': len(history['train_loss'])}
            results.append(result)
    
    # Put batch_size back into param_space for consistency
    param_space['batch_size'] = batch_sizes
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Check if the metric exists
    if metric not in results_df.columns:
        print(f"Warning: Metric '{metric}' not found. Available metrics: {list(results_df.columns)}")
        # Use iou_score as fallback
        if 'iou_score' in results_df.columns:
            metric = 'iou_score'
            print(f"Using '{metric}' instead.")
        else:
            # Use first available score metric
            score_metrics = [col for col in results_df.columns if 'score' in col]
            if score_metrics:
                metric = score_metrics[0]
                print(f"Using '{metric}' instead.")
            else:
                # Default to 'loss' if no score metric is available
                metric = 'loss'
                print(f"Using '{metric}' instead.")
    
    # Get best result based on specified metric
    best_idx = results_df[metric].argmax() if 'score' in metric else results_df[metric].argmin()
    best_result = results_df.iloc[best_idx].to_dict()
    
    print(f"\nGrid search completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best {metric}: {best_result[metric]:.4f}")
    
    return results_df, best_result


def perform_bayesian_optimization(
    param_space: Dict[str, Union[Tuple[float, float], List[Any]]],
    train_dataset,
    val_dataset,
    test_dataset,
    train_model_fn: Callable,
    get_model_fn: Callable,
    criterion_fn: Callable,
    device: torch.device,
    n_trials: int = 35,
    metric: str = 'iou_score',
    seed: int = 42,
    num_workers: int = 4
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform Bayesian optimization with batch size grouping.
    """
    # Store results
    results = []
    
    # Keep track of data loaders for each batch size
    data_loaders = {}
    
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
        
        # Get or create data loaders for this batch size
        batch_size = int(hyperparameters['batch_size'])
        if batch_size not in data_loaders:
            data_loaders[batch_size] = (
                torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=True),
                torch.utils.data.DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=True),
                torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            )
        
        train_loader, val_loader, test_loader = data_loaders[batch_size]
        
        # Run trial
        test_metrics, history = run_trial(
            hyperparameters,
            train_loader, val_loader, test_loader,
            train_model_fn, get_model_fn, criterion_fn, device
        )
        
        # Store results
        result = {**hyperparameters, **test_metrics, 'best_epoch': len(history['train_loss'])}
        results.append(result)
        
        # Check if metric exists
        if metric not in test_metrics:
            # Find an appropriate metric
            if 'iou_score' in test_metrics:
                return test_metrics['iou_score']
            else:
                # Use first available score metric
                score_metrics = [key for key in test_metrics.keys() if 'score' in key]
                if score_metrics:
                    return test_metrics[score_metrics[0]]
                else:
                    # Default to negative loss if no score metric is available
                    return -test_metrics['loss']
        
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
    
    return results_df, best_result, study


def plot_optimization_comparison(
    grid_results: pd.DataFrame,
    bayesian_results: pd.DataFrame,
    metric: str = 'iou_score',
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
    optimized_metric: str = 'iou_score',
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
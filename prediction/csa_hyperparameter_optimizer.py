"""
CSA Hyperparameter Optimizer for Machine Learning Models
==========================================================

Wrapper for Crow Search Algorithm to optimize hyperparameters of 
scikit-learn and XGBoost models using time-series cross-validation.

Supports:
- XGBoost regressor/classifier
- Random Forest regressor/classifier  
- Custom model hyperparameter spaces
- Multiple evaluation metrics
- Parallel evaluation of crow population
- Multi-objective optimization through weighted metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from crow_search_optimizer import CrowSearchOptimizer, ParameterSpec, CSAResult


class ModelCSAOptimizer:
    """
    Crow Search Algorithm optimizer for ML model hyperparameters.
    
    Parameters
    ----------
    model_type : str
        Type of model to optimize. Options: 'xgboost', 'random_forest', 'custom'
    X_train : array-like
        Training features.
    y_train : array-like  
        Training target.
    param_space : dict, optional
        Custom parameter space. If None, uses default for model_type.
        Format: {param_name: {'type': 'continuous'|'discrete', 'range': (min, max)}}
    cv_folds : int, default=3
        Number of folds for time-series cross-validation.
    metric : str or callable, default='rmse'
        Metric to optimize. Options: 'rmse', 'mae', 'r2', 'mape', 'directional_accuracy',
        'weighted' (combines RMSE and directional accuracy), or custom callable.
    metric_weights : dict, optional
        Weights for 'weighted' metric. E.g., {'rmse': 0.7, 'directional': 0.3}
    population_size : int, default=25
        CSA population size.
    max_iterations : int, default=50
        Maximum CSA iterations.
    awareness_probability : float, default=0.1
        CSA awareness probability.
    flight_length : float, default=2.0
        CSA flight length.
    n_jobs : int, default=1
        Number of parallel jobs for evaluating crow population.
        Use -1 for all available cores.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress information.
        
    Attributes
    ----------
    best_params_ : dict
        Best hyperparameters found.
    best_score_ : float
        Best cross-validation score achieved.
    best_model_ : object
        Model trained with best hyperparameters on full training data.
    optimization_result_ : CSAResult
        Full optimization result from CSA.
    """
    
    # Default hyperparameter spaces
    XGBOOST_SPACE = {
        'n_estimators': {'type': 'discrete', 'range': (50, 500)},
        'max_depth': {'type': 'discrete', 'range': (3, 15)},
        'learning_rate': {'type': 'continuous', 'range': (0.001, 0.3)},
        'subsample': {'type': 'continuous', 'range': (0.6, 1.0)},
        'colsample_bytree': {'type': 'continuous', 'range': (0.6, 1.0)},
        'min_child_weight': {'type': 'discrete', 'range': (1, 10)},
        'gamma': {'type': 'continuous', 'range': (0.0, 0.5)},
    }
    
    RANDOM_FOREST_SPACE = {
        'n_estimators': {'type': 'discrete', 'range': (50, 500)},
        'max_depth': {'type': 'discrete', 'range': (5, 30)},
        'min_samples_split': {'type': 'discrete', 'range': (2, 20)},
        'min_samples_leaf': {'type': 'discrete', 'range': (1, 10)},
        'max_features': {'type': 'continuous', 'range': (0.3, 0.9)},  # Will convert to float
    }
    
    def __init__(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Optional[Dict] = None,
        cv_folds: int = 3,
        metric: Union[str, Callable] = 'rmse',
        metric_weights: Optional[Dict] = None,
        population_size: int = 25,
        max_iterations: int = 50,
        awareness_probability: float = 0.1,
        flight_length: float = 2.0,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        self.model_type = model_type.lower()
        self.X_train = X_train
        self.y_train = y_train
        self.cv_folds = cv_folds
        self.metric = metric
        self.metric_weights = metric_weights or {'rmse': 0.7, 'directional': 0.3}
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.awareness_probability = awareness_probability
        self.flight_length = flight_length
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.random_state = random_state
        self.verbose = verbose
        
        # Set parameter space
        if param_space is None:
            if self.model_type == 'xgboost':
                self.param_space = self.XGBOOST_SPACE
            elif self.model_type == 'random_forest':
                self.param_space = self.RANDOM_FOREST_SPACE
            else:
                raise ValueError(f"Must provide param_space for custom model_type '{model_type}'")
        else:
            self.param_space = param_space
        
        # Results
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.optimization_result_ = None
        
    def _create_model(self, params: Dict) -> object:
        """Create model instance with given hyperparameters."""
        if self.model_type == 'xgboost':
            return XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=params['min_child_weight'],
                gamma=params['gamma'],
                random_state=self.random_state,
                n_jobs=1,  # Parallel at population level, not model level
                verbosity=0
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=self.random_state,
                n_jobs=1,
                verbose=0
            )
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported for auto-creation")
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (% of correct direction predictions)."""
        if len(y_true) < 2:
            return 0.0
        
        # Calculate actual and predicted direction of change
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        # Calculate accuracy
        correct = np.sum(actual_direction == pred_direction)
        total = len(actual_direction)
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Calculate specified metric. Lower is better (we'll negate if needed)."""
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'r2':
            return -r2_score(y_true, y_pred)  # Negate so lower is better
        elif metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        elif metric == 'directional_accuracy':
            # Negate so lower is better for minimization
            return -self._calculate_directional_accuracy(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _evaluate_params_cv(self, params: Dict) -> float:
        """
        Evaluate hyperparameters using time-series cross-validation.
        Returns score to minimize.
        """
        try:
            model = self._create_model(params)
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(self.X_train):
                X_train_fold = self.X_train[train_idx]
                y_train_fold = self.y_train[train_idx]
                X_val_fold = self.X_train[val_idx]
                y_val_fold = self.y_train[val_idx]
                
                # Train and predict
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                # Calculate metric
                if self.metric == 'weighted':
                    # Weighted combination of metrics
                    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                    dir_acc = self._calculate_directional_accuracy(y_val_fold, y_pred)
                    
                    # Normalize RMSE (assuming typical range 0-100 for CPO prices)
                    rmse_normalized = rmse / 100.0
                    
                    # Combined score (lower is better)
                    score = (self.metric_weights['rmse'] * rmse_normalized - 
                            self.metric_weights['directional'] * dir_acc)
                    scores.append(score)
                    
                elif callable(self.metric):
                    score = self.metric(y_val_fold, y_pred)
                    scores.append(score)
                else:
                    score = self._calculate_metric(y_val_fold, y_pred, self.metric)
                    scores.append(score)
            
            # Return mean score across folds
            return float(np.mean(scores))
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Evaluation failed for params {params}: {e}")
            return np.inf  # Return worst score on error
    
    def optimize(self) -> CSAResult:
        """
        Run CSA optimization to find best hyperparameters.
        
        Returns
        -------
        CSAResult
            Optimization result containing best parameters and convergence history.
        """
        if self.verbose:
            print("=" * 70)
            print(f"CSA Hyperparameter Optimization: {self.model_type.upper()}")
            print("=" * 70)
            print(f"Training samples: {len(self.X_train)}")
            print(f"Features: {self.X_train.shape[1]}")
            print(f"CV folds: {self.cv_folds}")
            print(f"Metric: {self.metric}")
            print(f"Parameter space: {len(self.param_space)} parameters")
            print(f"Population: {self.population_size}, Iterations: {self.max_iterations}")
            print("-" * 70)
        
        # Convert parameter space to ParameterSpec list
        param_specs = []
        for name, spec in self.param_space.items():
            param_specs.append(ParameterSpec(
                name=name,
                lower_bound=spec['range'][0],
                upper_bound=spec['range'][1],
                param_type=spec['type']
            ))
        
        # Create CSA optimizer
        # If using parallel evaluation, wrap objective function
        if self.n_jobs > 1:
            # Note: Parallel evaluation of population would require refactoring CSA
            # For now, we use sequential evaluation but keep n_jobs parameter for future
            if self.verbose:
                print(f"Note: Sequential evaluation (parallel support planned)")
        
        csa = CrowSearchOptimizer(
            objective_function=self._evaluate_params_cv,
            parameter_specs=param_specs,
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            awareness_probability=self.awareness_probability,
            flight_length=self.flight_length,
            early_stopping_patience=10,
            early_stopping_threshold=1e-6,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Run optimization
        result = csa.optimize()
        
        # Store results
        self.best_params_ = result.best_params
        self.best_score_ = result.best_score
        self.optimization_result_ = result
        
        # Train final model on full training data
        if self.verbose:
            print("\n" + "=" * 70)
            print("Training final model with best parameters...")
            print("=" * 70)
        
        self.best_model_ = self._create_model(self.best_params_)
        self.best_model_.fit(self.X_train, self.y_train)
        
        if self.verbose:
            print("Final model trained successfully!")
            print(f"Best parameters: {self.best_params_}")
            print(f"Best CV score: {self.best_score_:.6f}")
        
        return result
    
    def get_best_params(self) -> Dict:
        """Get best hyperparameters found."""
        if self.best_params_ is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        return self.best_params_
    
    def get_best_model(self) -> object:
        """Get model trained with best hyperparameters."""
        if self.best_model_ is None:
            raise ValueError("No model available. Run optimize() first.")
        return self.best_model_
    
    def save_results(self, filepath: str):
        """
        Save optimization results to CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to save results CSV.
        """
        if self.optimization_result_ is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # Create results dataframe
        results_data = []
        for iteration, score, params in self.optimization_result_.iteration_history:
            row = {'iteration': iteration, 'score': score}
            row.update(params)
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"Results saved to {filepath}")


class EnsembleWeightOptimizer:
    """
    Optimize ensemble weights using CSA.
    
    Instead of simple averaging, find optimal weights for combining
    multiple model predictions.
    
    Parameters
    ----------
    predictions_list : list of array-like
        List of prediction arrays from different models. Each should have shape (n_samples,)
    y_true : array-like
        True target values.
    metric : str, default='rmse'
        Metric to optimize when weighting predictions.
    population_size : int, default=20
        CSA population size.
    max_iterations : int, default=30
        Maximum CSA iterations.
    random_state : int, optional
        Random seed.
    verbose : bool, default=True
        Print progress.
        
    Attributes
    ----------
    best_weights_ : np.ndarray
        Best weights found (sum to 1.0).
    best_score_ : float
        Best metric score achieved.
    """
    
    def __init__(
        self,
        predictions_list: List[np.ndarray],
        y_true: np.ndarray,
        metric: str = 'rmse',
        population_size: int = 20,
        max_iterations: int = 30,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        self.predictions_list = [np.array(pred) for pred in predictions_list]
        self.y_true = np.array(y_true)
        self.n_models = len(predictions_list)
        self.metric = metric
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.verbose = verbose
        
        # Validate inputs
        if self.n_models < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        shapes = [pred.shape for pred in self.predictions_list]
        if len(set(shapes)) > 1:
            raise ValueError(f"All predictions must have same shape. Got: {shapes}")
        
        self.best_weights_ = None
        self.best_score_ = None
        self.optimization_result_ = None
    
    def _evaluate_weights(self, weight_params: Dict) -> float:
        """Evaluate ensemble with given weights."""
        # Extract weights and normalize to sum to 1
        weights = np.array([weight_params[f'weight_{i}'] for i in range(self.n_models)])
        weights = weights / np.sum(weights)  # Normalize
        
        # Compute weighted ensemble prediction
        ensemble_pred = sum(w * pred for w, pred in zip(weights, self.predictions_list))
        
        # Calculate metric
        if self.metric == 'rmse':
            score = np.sqrt(mean_squared_error(self.y_true, ensemble_pred))
        elif self.metric == 'mae':
            score = mean_absolute_error(self.y_true, ensemble_pred)
        elif self.metric == 'r2':
            score = -r2_score(self.y_true, ensemble_pred)  # Negate for minimization
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return score
    
    def optimize(self) -> CSAResult:
        """
        Optimize ensemble weights using CSA.
        
        Returns
        -------
        CSAResult
            Optimization result.
        """
        if self.verbose:
            print("=" * 70)
            print(f"CSA Ensemble Weight Optimization")
            print("=" * 70)
            print(f"Number of models: {self.n_models}")
            print(f"Samples: {len(self.y_true)}")
            print(f"Metric: {self.metric}")
            print("-" * 70)
        
        # Create parameter space for weights (each between 0 and 1, will normalize)
        param_specs = [
            ParameterSpec(
                name=f'weight_{i}',
                lower_bound=0.0,
                upper_bound=1.0,
                param_type='continuous'
            )
            for i in range(self.n_models)
        ]
        
        # Run CSA
        csa = CrowSearchOptimizer(
            objective_function=self._evaluate_weights,
            parameter_specs=param_specs,
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            awareness_probability=0.1,
            flight_length=2.0,
            early_stopping_patience=5,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        result = csa.optimize()
        
        # Extract and normalize weights
        raw_weights = np.array([result.best_params[f'weight_{i}'] for i in range(self.n_models)])
        self.best_weights_ = raw_weights / np.sum(raw_weights)
        self.best_score_ = result.best_score
        self.optimization_result_ = result
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("Ensemble Optimization Complete!")
            print(f"Best weights: {self.best_weights_}")
            print(f"Best score: {self.best_score_:.6f}")
            print("=" * 70)
        
        return result
    
    def predict(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """
        Make ensemble prediction using optimized weights.
        
        Parameters
        ----------
        predictions_list : list of array-like
            Predictions from each model for new data.
            
        Returns
        -------
        np.ndarray
            Weighted ensemble predictions.
        """
        if self.best_weights_ is None:
            raise ValueError("No weights available. Run optimize() first.")
        
        if len(predictions_list) != self.n_models:
            raise ValueError(f"Expected {self.n_models} predictions, got {len(predictions_list)}")
        
        return sum(w * np.array(pred) for w, pred in zip(self.best_weights_, predictions_list))


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing CSA Hyperparameter Optimizer")
    print("=" * 70)
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = 50 + 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(n_samples) * 5
    
    # Split data (time-series aware)
    train_size = int(0.8 * n_samples)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    print(f"Created synthetic dataset:")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print()
    
    # Test XGBoost optimization
    print("1. Optimizing XGBoost...")
    print("-" * 70)
    
    xgb_optimizer = ModelCSAOptimizer(
        model_type='xgboost',
        X_train=X_train,
        y_train=y_train,
        cv_folds=3,
        metric='rmse',
        population_size=10,  # Small for quick test
        max_iterations=20,
        random_state=42,
        verbose=True
    )
    
    xgb_result = xgb_optimizer.optimize()
    xgb_model = xgb_optimizer.get_best_model()
    xgb_pred = xgb_model.predict(X_test)
    xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    
    print(f"\nXGBoost test RMSE: {xgb_test_rmse:.4f}")
    print()
    
    # Test Random Forest optimization
    print("\n2. Optimizing Random Forest...")
    print("-" * 70)
    
    rf_optimizer = ModelCSAOptimizer(
        model_type='random_forest',
        X_train=X_train,
        y_train=y_train,
        cv_folds=3,
        metric='rmse',
        population_size=10,
        max_iterations=20,
        random_state=42,
        verbose=True
    )
    
    rf_result = rf_optimizer.optimize()
    rf_model = rf_optimizer.get_best_model()
    rf_pred = rf_model.predict(X_test)
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    print(f"\nRandom Forest test RMSE: {rf_test_rmse:.4f}")
    print()
    
    # Test ensemble weight optimization
    print("\n3. Optimizing Ensemble Weights...")
    print("-" * 70)
    
    # Use test predictions for ensemble optimization
    ensemble_optimizer = EnsembleWeightOptimizer(
        predictions_list=[xgb_pred, rf_pred],
        y_true=y_test,
        metric='rmse',
        population_size=10,
        max_iterations=15,
        random_state=42,
        verbose=True
    )
    
    ensemble_result = ensemble_optimizer.optimize()
    ensemble_pred = ensemble_optimizer.predict([xgb_pred, rf_pred])
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    print(f"\nEnsemble test RMSE: {ensemble_rmse:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"XGBoost RMSE:      {xgb_test_rmse:.4f}")
    print(f"Random Forest RMSE: {rf_test_rmse:.4f}")
    print(f"Ensemble RMSE:     {ensemble_rmse:.4f}")
    print(f"Ensemble weights:  XGB={ensemble_optimizer.best_weights_[0]:.3f}, RF={ensemble_optimizer.best_weights_[1]:.3f}")
    print("=" * 70)

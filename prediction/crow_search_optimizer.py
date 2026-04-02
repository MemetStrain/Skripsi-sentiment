"""
Crow Search Algorithm (CSA) Optimizer
======================================

Implementation of Crow Search Algorithm for optimization problems.
Inspired by the intelligent behavior of crows in finding and hiding food.

Reference:
Askarzadeh, A. (2016). A novel metaheuristic method for solving constrained 
engineering optimization problems: Crow search algorithm. Computers & Structures, 169, 1-12.

Key Concepts:
- Crows follow each other to discover hiding places
- Awareness probability (AP): probability that a crow detects being followed
- Flight length (fl): step size for searching
- Memory: each crow remembers its best position
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional, Union
import time
from dataclasses import dataclass, field


@dataclass
class ParameterSpec:
    """Specification for a parameter in the search space."""
    name: str
    lower_bound: float
    upper_bound: float
    param_type: str = 'continuous'  # 'continuous' or 'discrete'
    
    def __post_init__(self):
        if self.param_type not in ['continuous', 'discrete']:
            raise ValueError(f"param_type must be 'continuous' or 'discrete', got {self.param_type}")
        if self.lower_bound >= self.upper_bound:
            raise ValueError(f"lower_bound must be < upper_bound for {self.name}")


@dataclass
class CSAResult:
    """Result object from CSA optimization."""
    best_position: np.ndarray
    best_score: float
    best_params: Dict[str, Union[int, float]]
    convergence_history: List[float]
    iteration_history: List[Tuple[int, float, Dict]]
    total_iterations: int
    total_evaluations: int
    time_elapsed: float
    converged: bool = False
    convergence_iteration: Optional[int] = None


class CrowSearchOptimizer:
    """
    Crow Search Algorithm optimizer for continuous and discrete parameter spaces.
    
    Parameters
    ----------
    objective_function : callable
        Function to minimize. Should accept a dictionary of parameters and return a scalar score.
        Lower scores are better.
    parameter_specs : List[ParameterSpec]
        List of parameter specifications defining the search space.
    population_size : int, default=25
        Number of crows (candidate solutions) in the population.
    max_iterations : int, default=100
        Maximum number of iterations to run.
    awareness_probability : float, default=0.1
        Probability that a crow detects being followed (0 to 1).
        Higher values increase exploration, lower values increase exploitation.
    flight_length : float, default=2.0
        Controls the step size when following another crow.
        Higher values mean larger jumps in search space.
    early_stopping_patience : int, default=10
        Stop if no improvement for this many iterations.
    early_stopping_threshold : float, default=1e-6
        Minimum improvement to reset patience counter.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress information.
    
    Attributes
    ----------
    best_position_ : np.ndarray
        Best position found (normalized [0,1] space).
    best_score_ : float
        Best objective function value found.
    best_params_ : dict
        Best parameters in original scale.
    convergence_history_ : list
        Best score at each iteration.
    """
    
    def __init__(
        self,
        objective_function: Callable,
        parameter_specs: List[ParameterSpec],
        population_size: int = 25,
        max_iterations: int = 100,
        awareness_probability: float = 0.1,
        flight_length: float = 2.0,
        early_stopping_patience: int = 10,
        early_stopping_threshold: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        self.objective_function = objective_function
        self.parameter_specs = parameter_specs
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.awareness_probability = awareness_probability
        self.flight_length = flight_length
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Dimensionality of search space
        self.n_dims = len(parameter_specs)
        
        # Results attributes
        self.best_position_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.convergence_history_ = []
        self.iteration_history_ = []
        
    def _denormalize_position(self, normalized_pos: np.ndarray) -> Dict[str, Union[int, float]]:
        """Convert normalized position [0,1] to actual parameter values."""
        params = {}
        for i, spec in enumerate(self.parameter_specs):
            # Scale from [0,1] to [lower, upper]
            value = spec.lower_bound + normalized_pos[i] * (spec.upper_bound - spec.lower_bound)
            
            # Convert to discrete if needed
            if spec.param_type == 'discrete':
                value = int(np.round(value))
            
            params[spec.name] = value
        return params
    
    def _normalize_position(self, params: Dict[str, Union[int, float]]) -> np.ndarray:
        """Convert actual parameter values to normalized position [0,1]."""
        normalized = np.zeros(self.n_dims)
        for i, spec in enumerate(self.parameter_specs):
            value = params[spec.name]
            # Scale from [lower, upper] to [0,1]
            normalized[i] = (value - spec.lower_bound) / (spec.upper_bound - spec.lower_bound)
        return normalized
    
    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        """Ensure position stays within [0,1] bounds."""
        return np.clip(position, 0.0, 1.0)
    
    def _evaluate_position(self, position: np.ndarray) -> float:
        """Evaluate objective function at given position."""
        params = self._denormalize_position(position)
        try:
            score = self.objective_function(params)
            return score
        except Exception as e:
            if self.verbose:
                print(f"Warning: Evaluation failed with params {params}: {e}")
            return np.inf  # Return worst possible score on error
    
    def optimize(self) -> CSAResult:
        """
        Run the Crow Search Algorithm optimization.
        
        Returns
        -------
        CSAResult
            Object containing optimization results and history.
        """
        start_time = time.time()
        
        # Initialize population randomly in [0,1] space
        positions = np.random.uniform(0, 1, (self.population_size, self.n_dims))
        
        # Evaluate initial population
        fitness = np.array([self._evaluate_position(pos) for pos in positions])
        
        # Initialize memory (best position each crow has found)
        memory = positions.copy()
        memory_fitness = fitness.copy()
        
        # Track global best
        best_idx = np.argmin(memory_fitness)
        self.best_position_ = memory[best_idx].copy()
        self.best_score_ = memory_fitness[best_idx]
        self.best_params_ = self._denormalize_position(self.best_position_)
        
        self.convergence_history_ = [self.best_score_]
        self.iteration_history_ = [(0, self.best_score_, self.best_params_.copy())]
        
        # Early stopping tracking
        patience_counter = 0
        last_best_score = self.best_score_
        converged = False
        convergence_iteration = None
        
        if self.verbose:
            print(f"CSA Optimization Started")
            print(f"Population: {self.population_size}, Iterations: {self.max_iterations}")
            print(f"AP: {self.awareness_probability}, FL: {self.flight_length}")
            print(f"Initial best score: {self.best_score_:.6f}")
            print("-" * 60)
        
        # Main optimization loop
        total_evaluations = self.population_size  # Initial evaluations
        
        for iteration in range(1, self.max_iterations + 1):
            new_positions = positions.copy()
            
            for i in range(self.population_size):
                # Randomly select another crow to follow
                j = np.random.randint(0, self.population_size)
                while j == i:
                    j = np.random.randint(0, self.population_size)
                
                # Generate random number to check awareness
                r = np.random.uniform(0, 1)
                
                if r >= self.awareness_probability:
                    # Crow j doesn't know it's being followed - follow to its memory
                    # New position = current position + random_step * flight_length * (memory_j - current_position)
                    random_step = np.random.uniform(0, 1, self.n_dims)
                    new_positions[i] = positions[i] + random_step * self.flight_length * (memory[j] - positions[i])
                else:
                    # Crow j knows it's being followed - go to random position
                    new_positions[i] = np.random.uniform(0, 1, self.n_dims)
                
                # Ensure position stays in bounds
                new_positions[i] = self._clip_position(new_positions[i])
            
            # Evaluate new positions
            new_fitness = np.array([self._evaluate_position(pos) for pos in new_positions])
            total_evaluations += self.population_size
            
            # Update positions and memory
            for i in range(self.population_size):
                # If new position is better, accept it
                if new_fitness[i] < fitness[i]:
                    positions[i] = new_positions[i]
                    fitness[i] = new_fitness[i]
                    
                    # Update memory if this is the best position this crow has found
                    if new_fitness[i] < memory_fitness[i]:
                        memory[i] = new_positions[i]
                        memory_fitness[i] = new_fitness[i]
            
            # Update global best
            best_idx = np.argmin(memory_fitness)
            if memory_fitness[best_idx] < self.best_score_:
                improvement = self.best_score_ - memory_fitness[best_idx]
                self.best_position_ = memory[best_idx].copy()
                self.best_score_ = memory_fitness[best_idx]
                self.best_params_ = self._denormalize_position(self.best_position_)
                
                if self.verbose and iteration % 5 == 0:
                    print(f"Iter {iteration:3d}: New best = {self.best_score_:.6f} (improvement: {improvement:.6f})")
            
            # Track convergence
            self.convergence_history_.append(self.best_score_)
            self.iteration_history_.append((iteration, self.best_score_, self.best_params_.copy()))
            
            # Check early stopping
            if abs(self.best_score_ - last_best_score) < self.early_stopping_threshold:
                patience_counter += 1
            else:
                patience_counter = 0
            
            if patience_counter >= self.early_stopping_patience:
                converged = True
                convergence_iteration = iteration
                if self.verbose:
                    print(f"Early stopping at iteration {iteration} (no improvement for {self.early_stopping_patience} iterations)")
                break
            
            last_best_score = self.best_score_
            
            # Progress update
            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: Best = {self.best_score_:.6f}, Avg = {np.mean(fitness):.6f}")
        
        time_elapsed = time.time() - start_time
        
        if self.verbose:
            print("-" * 60)
            print(f"Optimization Complete!")
            print(f"Best score: {self.best_score_:.6f}")
            print(f"Best parameters: {self.best_params_}")
            print(f"Total iterations: {iteration}")
            print(f"Total evaluations: {total_evaluations}")
            print(f"Time elapsed: {time_elapsed:.2f}s")
        
        return CSAResult(
            best_position=self.best_position_,
            best_score=self.best_score_,
            best_params=self.best_params_,
            convergence_history=self.convergence_history_,
            iteration_history=self.iteration_history_,
            total_iterations=iteration,
            total_evaluations=total_evaluations,
            time_elapsed=time_elapsed,
            converged=converged,
            convergence_iteration=convergence_iteration
        )
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot convergence history.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot. If None, displays the plot.
        """
        if not self.convergence_history_:
            print("No convergence history available. Run optimize() first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history_, linewidth=2, color='#2E86AB')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Objective Value', fontsize=12)
        plt.title('CSA Convergence History', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
        plt.close()


# Test functions for validation
def sphere_function(params: Dict[str, float]) -> float:
    """Simple sphere function: f(x) = sum(x_i^2). Global minimum at origin."""
    return sum(v**2 for v in params.values())


def rastrigin_function(params: Dict[str, float]) -> float:
    """Rastrigin function: multimodal with many local minima. Global minimum at origin."""
    n = len(params)
    A = 10
    return A * n + sum(v**2 - A * np.cos(2 * np.pi * v) for v in params.values())


def rosenbrock_function(params: Dict[str, float]) -> float:
    """Rosenbrock function: banana-shaped valley. Global minimum at (1,1,...)."""
    values = list(params.values())
    return sum(100 * (values[i+1] - values[i]**2)**2 + (1 - values[i])**2 
               for i in range(len(values)-1))


if __name__ == "__main__":
    # Test CSA on simple optimization problems
    print("=" * 70)
    print("Testing Crow Search Algorithm")
    print("=" * 70)
    
    # Test 1: Sphere function (easy, unimodal)
    print("\n1. Testing on Sphere Function (2D)")
    print("-" * 70)
    param_specs = [
        ParameterSpec(name='x1', lower_bound=-5.0, upper_bound=5.0, param_type='continuous'),
        ParameterSpec(name='x2', lower_bound=-5.0, upper_bound=5.0, param_type='continuous')
    ]
    
    csa = CrowSearchOptimizer(
        objective_function=sphere_function,
        parameter_specs=param_specs,
        population_size=20,
        max_iterations=50,
        awareness_probability=0.1,
        flight_length=2.0,
        random_state=42,
        verbose=True
    )
    
    result = csa.optimize()
    print(f"\nExpected minimum: x1=0, x2=0, f(x)=0")
    print(f"Found minimum: {result.best_params}, f(x)={result.best_score:.6f}")
    csa.plot_convergence()
    
    # Test 2: Rastrigin function (harder, multimodal)
    print("\n\n2. Testing on Rastrigin Function (3D)")
    print("-" * 70)
    param_specs = [
        ParameterSpec(name='x1', lower_bound=-5.12, upper_bound=5.12, param_type='continuous'),
        ParameterSpec(name='x2', lower_bound=-5.12, upper_bound=5.12, param_type='continuous'),
        ParameterSpec(name='x3', lower_bound=-5.12, upper_bound=5.12, param_type='continuous')
    ]
    
    csa = CrowSearchOptimizer(
        objective_function=rastrigin_function,
        parameter_specs=param_specs,
        population_size=30,
        max_iterations=100,
        awareness_probability=0.1,
        flight_length=2.0,
        random_state=42,
        verbose=True
    )
    
    result = csa.optimize()
    print(f"\nExpected minimum: x1=0, x2=0, x3=0, f(x)=0")
    print(f"Found minimum: {result.best_params}, f(x)={result.best_score:.6f}")
    
    # Test 3: Discrete parameters
    print("\n\n3. Testing with Discrete Parameters")
    print("-" * 70)
    
    def discrete_test_function(params):
        """Simple function with discrete parameters."""
        target = {'n_trees': 100, 'max_depth': 10}
        return sum((params[k] - target[k])**2 for k in target)
    
    param_specs = [
        ParameterSpec(name='n_trees', lower_bound=50, upper_bound=200, param_type='discrete'),
        ParameterSpec(name='max_depth', lower_bound=5, upper_bound=20, param_type='discrete')
    ]
    
    csa = CrowSearchOptimizer(
        objective_function=discrete_test_function,
        parameter_specs=param_specs,
        population_size=15,
        max_iterations=30,
        awareness_probability=0.1,
        flight_length=2.0,
        random_state=42,
        verbose=True
    )
    
    result = csa.optimize()
    print(f"\nExpected minimum: n_trees=100, max_depth=10, f(x)=0")
    print(f"Found minimum: {result.best_params}, f(x)={result.best_score:.6f}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)

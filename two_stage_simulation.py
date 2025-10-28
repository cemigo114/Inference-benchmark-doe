"""
Two-Stage Design of Experiments (DOE) - Educational Demonstration

PURPOSE: Teaching tool for two-stage experimental design methodology

================================================================================
STAGE 1: SPACE-FILLING EXPLORATION
================================================================================
Goal: Efficiently explore the full parameter space

Method: Hybrid design for mixed categorical/numerical parameters
  - Pairwise coverage for categorical variables (deployment, scheduler, gaie_plugin_config)
  - Latin Hypercube Sampling (LHS) for numerical variables (replicas, concurrency, batch)

Result: ~30-50 well-distributed samples that cover:
  - All important pairwise categorical combinations
  - Space-filling coverage of numerical parameter ranges
  - Training data for Stage 2 surrogate model

================================================================================
STAGE 2: SURROGATE-BASED OPTIMIZATION
================================================================================
Goal: Exploit promising regions found in Stage 1

Method: Bayesian Optimization with Gaussian Process (GP) surrogate
  - Fit GP model on all Stage 1 data
  - Use Expected Improvement (EI) acquisition function
  - Sequentially select next experiments to maximize information gain

Result: Converge to optimal configuration faster than random/grid search

================================================================================
PERFORMANCE MODEL
================================================================================
Uses modelsim.py performance simulator with categorical configuration system

This version combines:
  - Simulator from two_stage_simulation.py (modelsim-based)
  - Optimizers from two_stage_simulation_v2.py (sophisticated DOE methods)

For real LLM-D benchmarks:
  - Stage 1: Generate pairwise+LHS treatment configurations, run benchmarks
  - Stage 2: Use real performance data to train GP, run BO-selected benchmarks
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
import itertools
from scipy.stats import qmc  # Latin Hypercube Sampling
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

import modelsim

# Set style for conference poster
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Mapping from categorical string values to integer indices for modelsim
DEPLOYMENT_MAP = {'standalone': 0, 'modelservice': 1}
SCHEDULER_MAP = {'none': 0, 'prefix': 1, 'kv': 2}


def simulate_performance_with_modelsim(config: Dict[str, Any], noise_level: float = 0.1) -> Dict[str, float]:
    """
    Wrapper for modelsim.simulate_performance that handles categorical config mapping.

    Parameters:
    - config: Dictionary with deployment, scheduler, gaie_plugin_config (ignored),
              replicas, concurrency, batch_size
    - noise_level: Standard deviation for Gaussian noise

    Returns:
    - Dictionary with throughput, latency, cost, efficiency metrics
    """
    # Extract parameters
    deployment = config.get('deployment', 'standalone')
    scheduler = config.get('scheduler', 'none')
    # gaie_plugin_config is not supported by modelsim, so we ignore it
    replicas = config.get('replicas', 1)
    concurrency = config.get('concurrency', 1)
    batch_size = config.get('batch_size', 1)

    # Map categorical strings to integer indices
    deployment_idx = DEPLOYMENT_MAP.get(deployment, 0)
    scheduler_idx = SCHEDULER_MAP.get(scheduler, 0)

    # Create base options and apply categorical settings
    base_options = modelsim.PerformanceOptions()
    options = modelsim.apply_categorical_settings(
        base_options,
        deployment=deployment_idx,
        scheduler=scheduler_idx
    )

    # Call modelsim.simulate_performance with numerical parameters
    params = {
        'replicas': replicas,
        'concurrency': concurrency,
        'batch_size': batch_size
    }

    results = modelsim.simulate_performance(params, options)

    # Calculate cost and efficiency with noise
    cost = replicas  # Cost = replicas as requested
    throughput = max(0, results['throughput'] * np.random.normal(1.0, noise_level))
    latency = max(0, results['latency'] * np.random.normal(1.0, noise_level))
    efficiency = throughput / (latency * cost) if latency > 0 and cost > 0 else 0

    # Return in format expected by rest of the code
    return {
        'throughput': throughput,
        'latency': latency,
        'cost': cost,
        'efficiency': efficiency
    }


class TwoStageOptimizer:
    """
    Implements proper two-stage Design of Experiments (DOE) methodology.

    Stage 1: Space-filling exploration (pairwise categorical + LHS numerical)
    Stage 2: Surrogate-based Bayesian optimization (GP + EI acquisition)
    """

    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
        self.evaluation_count = 0
        self.results = []

        # Parameter space definition (matches llm-d-benchmark)
        self.categorical_params = {
            'deployment': ['standalone', 'modelservice'],
            'scheduler': ['none', 'prefix', 'kv'],
            'gaie_plugin_config': ['default', 'prefix-cache-estimate-config', 'prefix-cache-tracking-config']
        }

        self.numerical_params = {
            'replicas': [1, 2, 4, 8],
            'concurrency': [1, 8, 32, 64, 128, 256],
            'batch_size': [1, 2, 4, 8, 16, 32]
        }

        # For GP encoding
        self.encoders = {
            param: LabelEncoder().fit(values)
            for param, values in self.categorical_params.items()
        }

    def _simulate_performance(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Simulate performance and track evaluation count.

        Parameters:
        - config: Dictionary with deployment, scheduler, gaie_plugin_config,
                 replicas, concurrency, batch_size

        Returns:
        - Dictionary with throughput, latency, cost, efficiency metrics
        """
        self.evaluation_count += 1
        return simulate_performance_with_modelsim(config, self.noise_level)

    def stage1_categorical_screening(self) -> List[Dict]:
        """
        Stage 1: Categorical screening with FIXED numerical parameters.

        Method:
        1. Generate pairwise coverage of categorical parameters
        2. Fix numerical parameters at reasonable defaults (middle values)
        3. Evaluate each categorical combination once
        4. Result: ~6-8 evaluations to identify best categorical config(s)

        Returns:
            List of evaluated configurations with performance data
        """
        print(f"Stage 1: Categorical Screening (Fixed Numerical Parameters)")

        # Generate pairwise coverage of categorical parameters
        categorical_configs = self._generate_pairwise_categorical()
        print(f"  Generated {len(categorical_configs)} categorical configurations")

        # Fixed numerical parameters (middle values)
        # FIX: Changed concurrency from 64 to 16 to avoid overload
        # With replicas=4, optimal concurrency = 4*4 = 16
        fixed_numerical = {
            'replicas': 4,           # Middle value of [1, 2, 4, 8]
            'concurrency': 16,       # FIXED: was 64, caused 400% overload!
            'batch_size': 8          # Middle value of [1, 2, 4, 8, 16, 32]
        }
        print(f"  Fixed numerical params: replicas={fixed_numerical['replicas']}, "
              f"concurrency={fixed_numerical['concurrency']}, "
              f"batch_size={fixed_numerical['batch_size']}")

        # Evaluate each categorical configuration
        stage1_results = []

        for cat_idx, cat_config in enumerate(categorical_configs):
            # Combine categorical and fixed numerical parameters
            full_config = {**cat_config, **fixed_numerical}

            # Evaluate
            performance = self._simulate_performance(full_config)
            performance['config'] = full_config
            performance['stage'] = 'stage1'
            performance['categorical_config'] = cat_config  # Track categorical separately
            stage1_results.append(performance)

            print(f"  Config {cat_idx+1}: {cat_config} -> Efficiency={performance['efficiency']:.3f}")

        print(f"  Total Stage 1 evaluations: {len(stage1_results)}")
        return stage1_results

    def _generate_pairwise_categorical(self) -> List[Dict]:
        """
        Generate pairwise (all-pairs) coverage of categorical parameters.

        Ensures every 2-way combination is tested at least once.

        For 3 factors with levels [2, 3, 3]:
        - deployment × scheduler: 2×3 = 6 pairs
        - deployment × gaie_plugin_config: 2×3 = 6 pairs
        - scheduler × gaie_plugin_config: 3×3 = 9 pairs
        Total: 21 unique pairs to cover

        Uses greedy algorithm to find minimal covering array.
        """
        deployments = self.categorical_params['deployment']
        schedulers = self.categorical_params['scheduler']
        gaie_configs = self.categorical_params['gaie_plugin_config']

        # Track which pairs have been covered
        uncovered_pairs = set()

        # Generate all required pairwise combinations
        for d in deployments:
            for s in schedulers:
                uncovered_pairs.add(('deployment', d, 'scheduler', s))

        for d in deployments:
            for g in gaie_configs:
                uncovered_pairs.add(('deployment', d, 'gaie_plugin_config', g))

        for s in schedulers:
            for g in gaie_configs:
                uncovered_pairs.add(('scheduler', s, 'gaie_plugin_config', g))

        configs = []

        # Greedy selection: pick configuration that covers most uncovered pairs
        while uncovered_pairs:
            best_config = None
            best_coverage = 0

            # Try all possible configurations
            for deployment in deployments:
                for scheduler in schedulers:
                    for gaie_config in gaie_configs:
                        config = {
                            'deployment': deployment,
                            'scheduler': scheduler,
                            'gaie_plugin_config': gaie_config
                        }

                        # Count how many uncovered pairs this config covers
                        pairs_covered = set([
                            ('deployment', deployment, 'scheduler', scheduler),
                            ('deployment', deployment, 'gaie_plugin_config', gaie_config),
                            ('scheduler', scheduler, 'gaie_plugin_config', gaie_config),
                        ])

                        coverage = len(pairs_covered & uncovered_pairs)

                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_config = config

            # Add best configuration
            configs.append(best_config)

            # Remove covered pairs
            pairs_covered_by_best = set([
                ('deployment', best_config['deployment'], 'scheduler', best_config['scheduler']),
                ('deployment', best_config['deployment'], 'gaie_plugin_config', best_config['gaie_plugin_config']),
                ('scheduler', best_config['scheduler'], 'gaie_plugin_config', best_config['gaie_plugin_config']),
            ])
            uncovered_pairs -= pairs_covered_by_best

        return configs

    def _lhs_sample_numerical(self, n_samples: int) -> List[Dict]:
        """
        Generate Latin Hypercube samples for numerical parameters.

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of numerical parameter configurations
        """
        # LHS sampler for 3 numerical dimensions
        sampler = qmc.LatinHypercube(d=3, seed=42)
        samples = sampler.random(n=n_samples)

        # Map [0,1] samples to discrete parameter values
        configs = []
        for sample in samples:
            # Map each dimension to discrete choices
            replica_idx = int(sample[0] * len(self.numerical_params['replicas']))
            replica_idx = min(replica_idx, len(self.numerical_params['replicas']) - 1)

            concurrency_idx = int(sample[1] * len(self.numerical_params['concurrency']))
            concurrency_idx = min(concurrency_idx, len(self.numerical_params['concurrency']) - 1)

            batch_idx = int(sample[2] * len(self.numerical_params['batch_size']))
            batch_idx = min(batch_idx, len(self.numerical_params['batch_size']) - 1)

            config = {
                'replicas': self.numerical_params['replicas'][replica_idx],
                'concurrency': self.numerical_params['concurrency'][concurrency_idx],
                'batch_size': self.numerical_params['batch_size'][batch_idx]
            }
            configs.append(config)

        return configs

    def stage2_bayesian_optimization(self, stage1_results: List[Dict],
                                      n_iterations: int = 20, top_k: int = 1) -> List[Dict]:
        """
        Stage 2: Bayesian Optimization for NUMERICAL parameters only.

        Method:
        1. Select top K categorical configurations from Stage 1
        2. For each top categorical config:
           - FIX the categorical parameters
           - Use BO to optimize ONLY numerical parameters (replicas, concurrency, batch)
           - Train GP on numerical parameter space only
        3. Use Expected Improvement (EI) acquisition function

        Args:
            stage1_results: Results from Stage 1 categorical screening
            n_iterations: Number of BO iterations per categorical config
            top_k: Number of top categorical configs to optimize (default: 1)

        Returns:
            List of Stage 2 evaluation results
        """
        print(f"\nStage 2: Bayesian Optimization (Numerical Parameters Only)")

        # Sort Stage 1 results by efficiency and select top K
        sorted_results = sorted(stage1_results, key=lambda x: x['efficiency'], reverse=True)
        top_categorical_configs = [r['categorical_config'] for r in sorted_results[:top_k]]

        print(f"  Optimizing top {top_k} categorical configuration(s) from Stage 1")

        all_stage2_results = []
        global_best_efficiency = -np.inf
        global_best_config = None

        # Optimize each top categorical configuration
        for cat_idx, fixed_categorical in enumerate(top_categorical_configs):
            print(f"\n  === Optimizing Categorical Config {cat_idx+1}/{top_k} ===")
            print(f"  Fixed: {fixed_categorical}")

            # Initial exploration: LHS sample numerical space for this categorical config
            initial_samples = self._generate_initial_numerical_samples(
                fixed_categorical, n_samples=5
            )

            # Evaluate initial samples
            local_results = []
            for sample in initial_samples:
                performance = self._simulate_performance(sample)
                performance['config'] = sample
                performance['stage'] = 'stage2'
                performance['categorical_config'] = fixed_categorical
                local_results.append(performance)
                all_stage2_results.append(performance)

            # Prepare GP training data (numerical parameters only)
            X_train, y_train = self._prepare_numerical_gp_data(local_results)
            all_X = X_train.copy()
            all_y = y_train.copy()

            # Track best for this categorical config
            best_idx = np.argmax(all_y)
            best_efficiency = all_y[best_idx]
            best_config = local_results[best_idx]['config']
            print(f"  Initial best: Efficiency={best_efficiency:.3f}")

            # Bayesian optimization loop (numerical parameters only)
            for iteration in range(n_iterations):
                # Fit GP on numerical parameters only
                gp = self._fit_gp_surrogate(all_X, all_y)

                # Generate candidate pool: fixed categorical, varied numerical
                candidates = self._generate_numerical_candidate_pool(
                    fixed_categorical, n_candidates=100
                )

                # Select next point using EI
                next_config = self._select_next_by_ei_numerical(
                    gp, candidates, all_X, all_y
                )

                # Evaluate
                performance = self._simulate_performance(next_config)
                performance['config'] = next_config
                performance['stage'] = 'stage2'
                performance['categorical_config'] = fixed_categorical
                local_results.append(performance)
                all_stage2_results.append(performance)

                # Update GP training data
                X_new = self._numerical_config_to_vector(next_config)
                y_new = performance['efficiency']
                all_X = np.vstack([all_X, X_new])
                all_y = np.append(all_y, y_new)

                # Update best
                if y_new > best_efficiency:
                    best_efficiency = y_new
                    best_config = next_config
                    print(f"  Iteration {iteration+1}/{n_iterations}: New best! Efficiency={best_efficiency:.3f}")
                else:
                    print(f"  Iteration {iteration+1}/{n_iterations}: Efficiency={y_new:.3f}")

            # Update global best
            if best_efficiency > global_best_efficiency:
                global_best_efficiency = best_efficiency
                global_best_config = best_config

            print(f"  Final best for this categorical: Efficiency={best_efficiency:.3f}")

        print(f"\n  === Stage 2 Complete ===")
        print(f"  Global best: Efficiency={global_best_efficiency:.3f}")
        print(f"  Config: {global_best_config}")

        return all_stage2_results

    def _generate_initial_numerical_samples(self, fixed_categorical: Dict,
                                             n_samples: int = 5) -> List[Dict]:
        """
        Generate initial LHS samples for numerical parameters with fixed categorical.
        """
        numerical_samples = self._lhs_sample_numerical(n_samples)
        full_configs = []
        for num_sample in numerical_samples:
            full_config = {**fixed_categorical, **num_sample}
            full_configs.append(full_config)
        return full_configs

    def _prepare_numerical_gp_data(self, results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare GP training data using ONLY numerical parameters.

        Categorical parameters are fixed in Stage 2, so we only encode numerical params.
        """
        X = []
        y = []

        for result in results:
            config = result['config']
            X.append(self._numerical_config_to_vector(config))
            y.append(result['efficiency'])

        return np.array(X), np.array(y)

    def _numerical_config_to_vector(self, config: Dict) -> np.ndarray:
        """
        Convert numerical parameters to feature vector (3D: replicas, concurrency, batch).

        Categorical parameters are ignored in Stage 2.
        """
        replica_idx = self.numerical_params['replicas'].index(config['replicas'])
        concurrency_idx = self.numerical_params['concurrency'].index(config['concurrency'])
        batch_idx = self.numerical_params['batch_size'].index(config['batch_size'])

        return np.array([replica_idx, concurrency_idx, batch_idx])

    def _generate_numerical_candidate_pool(self, fixed_categorical: Dict,
                                            n_candidates: int = 100) -> List[Dict]:
        """
        Generate candidate pool with FIXED categorical and VARIED numerical parameters.
        """
        candidates = []

        for _ in range(n_candidates):
            numerical_config = {
                'replicas': np.random.choice(self.numerical_params['replicas']),
                'concurrency': np.random.choice(self.numerical_params['concurrency']),
                'batch_size': np.random.choice(self.numerical_params['batch_size']),
            }
            full_config = {**fixed_categorical, **numerical_config}
            candidates.append(full_config)

        return candidates

    def _select_next_by_ei_numerical(self, gp: GaussianProcessRegressor, candidates: List[Dict],
                                      X_observed: np.ndarray, y_observed: np.ndarray) -> Dict:
        """
        Select next configuration using EI on numerical parameters only.
        """
        from scipy.stats import norm

        # Convert candidates to numerical feature vectors
        X_candidates = np.array([self._numerical_config_to_vector(c) for c in candidates])

        # GP predictions
        mu, sigma = gp.predict(X_candidates, return_std=True)

        # Current best
        f_best = np.max(y_observed)

        # Expected Improvement
        with np.errstate(divide='warn'):
            improvement = mu - f_best
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        # Select candidate with highest EI
        best_idx = np.argmax(ei)
        return candidates[best_idx]

    def _prepare_gp_data(self, results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for GP from evaluation results.

        Converts configurations to feature vectors and extracts efficiency targets.
        """
        X = []
        y = []

        for result in results:
            config = result['config']
            X.append(self._config_to_vector(config))
            y.append(result['efficiency'])

        return np.array(X), np.array(y)

    def _config_to_vector(self, config: Dict) -> np.ndarray:
        """
        Convert configuration dict to numeric feature vector for GP.

        Categorical parameters are label-encoded.
        Numerical parameters are converted to indices.
        """
        # Encode categorical parameters
        deployment_enc = self.encoders['deployment'].transform([config['deployment']])[0]
        scheduler_enc = self.encoders['scheduler'].transform([config['scheduler']])[0]
        gaie_enc = self.encoders['gaie_plugin_config'].transform([config['gaie_plugin_config']])[0]

        # Encode numerical parameters as indices
        replica_idx = self.numerical_params['replicas'].index(config['replicas'])
        concurrency_idx = self.numerical_params['concurrency'].index(config['concurrency'])
        batch_idx = self.numerical_params['batch_size'].index(config['batch_size'])

        return np.array([deployment_enc, scheduler_enc, gaie_enc,
                        replica_idx, concurrency_idx, batch_idx])

    def _fit_gp_surrogate(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """
        Fit Gaussian Process surrogate model.

        Uses Matern kernel which is good for non-smooth functions.
        """
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        gp.fit(X, y)
        return gp

    def _generate_candidate_pool(self, n_candidates: int = 100) -> List[Dict]:
        """
        Generate a pool of candidate configurations for acquisition function.

        Uses random sampling over the discrete parameter space.
        """
        candidates = []

        for _ in range(n_candidates):
            config = {
                'deployment': np.random.choice(self.categorical_params['deployment']),
                'scheduler': np.random.choice(self.categorical_params['scheduler']),
                'gaie_plugin_config': np.random.choice(self.categorical_params['gaie_plugin_config']),
                'replicas': np.random.choice(self.numerical_params['replicas']),
                'concurrency': np.random.choice(self.numerical_params['concurrency']),
                'batch_size': np.random.choice(self.numerical_params['batch_size']),
            }
            candidates.append(config)

        return candidates

    def _select_next_by_ei(self, gp: GaussianProcessRegressor, candidates: List[Dict],
                           X_observed: np.ndarray, y_observed: np.ndarray) -> Dict:
        """
        Select next configuration using Expected Improvement acquisition function.

        EI(x) = E[max(f(x) - f_best, 0)]

        where f_best is the current best observed value.
        """
        from scipy.stats import norm

        # Convert candidates to feature vectors
        X_candidates = np.array([self._config_to_vector(c) for c in candidates])

        # GP predictions
        mu, sigma = gp.predict(X_candidates, return_std=True)

        # Current best
        f_best = np.max(y_observed)

        # Expected Improvement
        with np.errstate(divide='warn'):
            improvement = mu - f_best
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0  # Handle zero sigma

        # Select candidate with highest EI
        best_idx = np.argmax(ei)
        return candidates[best_idx]


class BaselineComparator:
    """
    Implements baseline methods for comparison.
    """

    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
        self.evaluation_count = 0

        # Parameter space definition (same as TwoStageOptimizer)
        self.categorical_params = {
            'deployment': ['standalone', 'modelservice'],
            'scheduler': ['none', 'prefix', 'kv'],
            'gaie_plugin_config': ['default', 'prefix-cache-estimate-config', 'prefix-cache-tracking-config']
        }

    def _simulate_performance(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Simulate performance using shared simulation function.

        Parameters:
        - config: Dictionary with deployment, scheduler, gaie_plugin_config,
                 replicas, concurrency, batch_size

        Returns:
        - Dictionary with throughput, latency, cost, efficiency metrics
        """
        self.evaluation_count += 1
        return simulate_performance_with_modelsim(config, self.noise_level)

    def _generate_pairwise_categorical(self) -> List[Dict]:
        """
        Generate pairwise coverage of categorical parameters (same as TwoStageOptimizer).
        """
        deployments = self.categorical_params['deployment']
        schedulers = self.categorical_params['scheduler']
        gaie_configs = self.categorical_params['gaie_plugin_config']

        uncovered_pairs = set()

        for d in deployments:
            for s in schedulers:
                uncovered_pairs.add(('deployment', d, 'scheduler', s))

        for d in deployments:
            for g in gaie_configs:
                uncovered_pairs.add(('deployment', d, 'gaie_plugin_config', g))

        for s in schedulers:
            for g in gaie_configs:
                uncovered_pairs.add(('scheduler', s, 'gaie_plugin_config', g))

        configs = []

        while uncovered_pairs:
            best_config = None
            best_coverage = 0

            for deployment in deployments:
                for scheduler in schedulers:
                    for gaie_config in gaie_configs:
                        config = {
                            'deployment': deployment,
                            'scheduler': scheduler,
                            'gaie_plugin_config': gaie_config
                        }

                        pairs_covered = set([
                            ('deployment', deployment, 'scheduler', scheduler),
                            ('deployment', deployment, 'gaie_plugin_config', gaie_config),
                            ('scheduler', scheduler, 'gaie_plugin_config', gaie_config),
                        ])

                        coverage = len(pairs_covered & uncovered_pairs)

                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_config = config

            configs.append(best_config)

            pairs_covered_by_best = set([
                ('deployment', best_config['deployment'], 'scheduler', best_config['scheduler']),
                ('deployment', best_config['deployment'], 'gaie_plugin_config', best_config['gaie_plugin_config']),
                ('scheduler', best_config['scheduler'], 'gaie_plugin_config', best_config['gaie_plugin_config']),
            ])
            uncovered_pairs -= pairs_covered_by_best

        return configs

    def random_search(self, total_evaluations: int) -> List[Dict]:
        """Random search baseline."""
        print(f"\nRandom Search Baseline ({total_evaluations} evaluations)")

        results = []
        categorical_options = [
            {'deployment': 'standalone', 'scheduler': 'none', 'gaie_plugin_config': 'default'},
            {'deployment': 'standalone', 'scheduler': 'prefix', 'gaie_plugin_config': 'prefix-cache-estimate-config'},
            {'deployment': 'modelservice', 'scheduler': 'none', 'gaie_plugin_config': 'default'},
            {'deployment': 'modelservice', 'scheduler': 'kv', 'gaie_plugin_config': 'prefix-cache-tracking-config'},
        ]

        for i in range(total_evaluations):
            config = np.random.choice(categorical_options).copy()
            config['replicas'] = np.random.choice([1, 2, 4, 8])
            config['concurrency'] = np.random.choice([1, 8, 32, 64, 128, 256])
            config['batch_size'] = np.random.choice([1, 2, 4, 8, 16, 32])

            performance = self._simulate_performance(config)
            performance['config'] = config
            performance['stage'] = 'random'
            results.append(performance)

        return results

    def grid_search(self, max_configs: int = 50) -> List[Dict]:
        """
        Grid search baseline - IMPROVED VERSION.

        Uses same categorical coverage as Stage 1 (9 pairwise combos)
        and proper numerical grid including optimal values.
        """
        print(f"\nGrid Search Baseline ({max_configs} evaluations)")

        results = []

        # FIX: Use same categorical coverage as Two-Stage Stage 1
        # This makes comparison fair - both methods explore same categorical space
        categorical_options = self._generate_pairwise_categorical()

        # FIX: Expanded numerical grid to include optimal values
        numerical_values = {
            'replicas': [2, 4, 8],              # Include more values
            'concurrency': [8, 16, 32, 64],     # CRITICAL: Include 8 (optimal!)
            'batch_size': [4, 8, 16]            # Include more values
        }

        print(f"  Categorical combos: {len(categorical_options)}")
        print(f"  Numerical grid: {len(numerical_values['replicas'])} × "
              f"{len(numerical_values['concurrency'])} × "
              f"{len(numerical_values['batch_size'])} = "
              f"{len(numerical_values['replicas']) * len(numerical_values['concurrency']) * len(numerical_values['batch_size'])} points")

        # Sample uniformly from grid until budget exhausted
        import itertools
        all_combinations = []
        for cat_config in categorical_options:
            for replicas in numerical_values['replicas']:
                for concurrency in numerical_values['concurrency']:
                    for batch_size in numerical_values['batch_size']:
                        config = cat_config.copy()
                        config.update({
                            'replicas': replicas,
                            'concurrency': concurrency,
                            'batch_size': batch_size
                        })
                        all_combinations.append(config)

        # Shuffle and take first max_configs
        import random
        random.seed(42)
        random.shuffle(all_combinations)

        for config in all_combinations[:max_configs]:
            performance = self._simulate_performance(config)
            performance['config'] = config
            performance['stage'] = 'grid'
            results.append(performance)

        return results


def run_comparison_study(fast_demo=False):
    """
    Run the complete comparison study.

    Args:
        fast_demo: If True, reduce iterations for faster demo (default: False)
    """
    print("=" * 80)
    print("TWO-STAGE DESIGN OF EXPERIMENTS SIMULATION")
    print("LLM Infrastructure Benchmarking")
    if fast_demo:
        print("(FAST DEMO MODE)")
    print("=" * 80)

    # Set seed for reproducibility
    np.random.seed(42)

    # Two-stage DOE optimization
    # IMPROVED: Increased budget to properly demonstrate BO advantage
    # Fast: 50 evals, Full: 100 evals (vs original 10/20)
    n_stage2_iterations = 50 if fast_demo else 100  # BO iterations for numerical params
    top_k_categorical = 1 if fast_demo else 1  # Number of top categorical configs to optimize

    # IMPROVED: Reduced noise for clearer signal
    optimizer = TwoStageOptimizer(noise_level=0.05)

    # Stage 1: Categorical screening with FIXED numerical parameters
    stage1_results = optimizer.stage1_categorical_screening()

    # Stage 2: Bayesian optimization for NUMERICAL parameters only
    # (categorical parameters are FIXED to top K from Stage 1)
    stage2_results = optimizer.stage2_bayesian_optimization(
        stage1_results,
        n_iterations=n_stage2_iterations,
        top_k=top_k_categorical
    )

    two_stage_results = stage1_results + stage2_results

    # Baseline comparisons
    # IMPROVED: Match noise level for fair comparison
    comparator = BaselineComparator(noise_level=0.05)
    random_results = comparator.random_search(total_evaluations=len(two_stage_results))
    grid_results = comparator.grid_search(max_configs=len(two_stage_results))

    # Analysis and visualization
    analyze_results(two_stage_results, random_results, grid_results)

    return {
        'two_stage': two_stage_results,
        'random': random_results,
        'grid': grid_results,
        'evaluation_count': optimizer.evaluation_count
    }


def analyze_results(two_stage_results: List[Dict], random_results: List[Dict],
                   grid_results: List[Dict]):
    """
    Analyze and visualize results.
    """
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    # Convert to DataFrames
    methods = ['Two-Stage', 'Random', 'Grid']
    results_list = [two_stage_results, random_results, grid_results]

    all_results = []
    for method, results in zip(methods, results_list):
        for result in results:
            row = result.copy()
            row['method'] = method
            all_results.append(row)

    df = pd.DataFrame(all_results)

    # Summary statistics
    print("\nMethod Comparison:")
    summary = df.groupby('method').agg({
        'throughput': ['mean', 'max', 'std'],
        'latency': ['mean', 'min', 'std'],
        'efficiency': ['mean', 'max', 'std']
    }).round(2)
    print(summary)

    # Best configurations found
    print("\nBest Configuration by Method:")
    for method in methods:
        method_df = df[df['method'] == method]
        best_idx = method_df['efficiency'].idxmax()
        best_config = method_df.loc[best_idx]

        print(f"\n{method}:")
        print(f"  Throughput: {best_config['throughput']:.1f}")
        print(f"  Latency: {best_config['latency']:.1f}")
        print(f"  Efficiency: {best_config['efficiency']:.3f}")
        print(f"  Config: {best_config['config']}")

    # Create visualizations
    create_visualizations(df)


def create_visualizations(df: pd.DataFrame):
    """
    Create visualizations for conference poster.
    """
    print("\nCreating visualizations...")

    # DEFINE CONSISTENT COLORS FOR ALL PLOTS
    COLORS = {
        'Two-Stage': '#1f77b4',  # Blue
        'Random': '#ff7f0e',     # Orange
        'Grid': '#2ca02c'        # Green
    }
    methods_order = ['Two-Stage', 'Random', 'Grid']

    # Set up the plotting style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Performance Comparison
    ax1 = plt.subplot(2, 3, 1)
    throughput_data = [df[df['method'] == method]['throughput'] for method in methods_order]
    bp1 = ax1.boxplot(throughput_data, labels=methods_order, patch_artist=True)
    ax1.set_title('Throughput Distribution', fontweight='bold')
    ax1.set_ylabel('Throughput (requests/sec)')
    ax1.grid(True, alpha=0.3)

    # Color the boxes with CONSISTENT colors
    for patch, method in zip(bp1['boxes'], methods_order):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)

    # 2. Efficiency Comparison
    ax2 = plt.subplot(2, 3, 2)
    efficiency_data = [df[df['method'] == method]['efficiency'] for method in methods_order]
    bp2 = ax2.boxplot(efficiency_data, labels=methods_order, patch_artist=True)
    ax2.set_title('Efficiency Distribution', fontweight='bold')
    ax2.set_ylabel('Efficiency (throughput/latency/cost)')
    ax2.grid(True, alpha=0.3)

    for patch, method in zip(bp2['boxes'], methods_order):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)

    # 3. Optimization Progress
    ax3 = plt.subplot(2, 3, 3)
    for method in methods_order:
        method_df = df[df['method'] == method].sort_values('efficiency')
        ax3.plot(range(len(method_df)), method_df['efficiency'],
                marker='o', label=method, linewidth=2, markersize=4,
                color=COLORS[method])

    ax3.set_title('Optimization Progress', fontweight='bold')
    ax3.set_xlabel('Evaluation Number')
    ax3.set_ylabel('Best Efficiency Found')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Throughput vs Latency Trade-off
    ax4 = plt.subplot(2, 3, 4)
    for method in methods_order:
        method_df = df[df['method'] == method]
        ax4.scatter(method_df['latency'], method_df['throughput'],
                   label=method, alpha=0.7, s=50, color=COLORS[method])

    ax4.set_title('Throughput vs Latency Trade-off', fontweight='bold')
    ax4.set_xlabel('Latency (ms)')
    ax4.set_ylabel('Throughput (requests/sec)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # set xlim max to max of values that are < 10000ms
    ax4.set_xlim(0, df['latency'][df['latency'] < 10000].max() * 1.1)

    # 5. Parameter Space Coverage (simplified)
    ax5 = plt.subplot(2, 3, 5)
    two_stage_df = df[df['method'] == 'Two-Stage']

    # Extract numerical parameters safely
    replicas = []
    concurrency = []
    for config in two_stage_df['config']:
        if isinstance(config, dict):
            replicas.append(config.get('replicas', 1))
            # Handle both max_concurrency and concurrency keys
            concurrency.append(config.get('max_concurrency', config.get('concurrency', 1)))
        else:
            replicas.append(1)
            concurrency.append(1)

    if replicas:  # Only plot if we have data
        scatter = ax5.scatter(replicas, concurrency,
                             c=two_stage_df['efficiency'],
                             cmap='viridis', s=100, alpha=0.7)
        ax5.set_title('Two-Stage Parameter Exploration', fontweight='bold')
        ax5.set_xlabel('Replicas')
        ax5.set_ylabel('Concurrency')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Efficiency')

    # 6. Method Comparison Summary
    ax6 = plt.subplot(2, 3, 6)
    summary_stats = df.groupby('method').agg({
        'throughput': 'max',
        'efficiency': 'max',
        'latency': 'min'
    })

    # Normalize for comparison
    normalized_stats = summary_stats.div(summary_stats.max())

    # Reorder to match methods_order
    normalized_stats = normalized_stats.reindex(methods_order)

    x = np.arange(len(normalized_stats.columns))
    width = 0.25

    for i, method in enumerate(methods_order):
        ax6.bar(x + i*width, normalized_stats.loc[method],
               width, label=method, alpha=0.8, color=COLORS[method])

    ax6.set_title('Normalized Best Performance', fontweight='bold')
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Normalized Score')
    ax6.set_xticks(x + width)
    ax6.set_xticklabels(['Max Throughput', 'Max Efficiency', 'Min Latency'])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('poster_visualizations_v3.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print("Visualizations saved to poster_visualizations_v3.png")


if __name__ == "__main__":
    import sys

    # Check for fast demo mode
    fast_demo = '--fast' in sys.argv or '--demo' in sys.argv

    results = run_comparison_study(fast_demo=fast_demo)
    print(f"\nSimulation completed with {results['evaluation_count']} total evaluations")
    print(f"\nUsage: python two_stage_simulation_v3.py [--fast|--demo]")

"""
Treatment generation strategies.

Implements various sampling strategies for generating treatments:
- Latin Hypercube Sampling (LHS)
- Sobol sequences
- Random sampling
- Factorial designs
- Grid search
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import itertools
import random
import numpy as np
from scipy.stats import qmc

from .parameter_space import ParameterSpace, Parameter, ParameterType
from .treatment import Treatment


class GenerationStrategy(ABC):
    """Base class for treatment generation strategies."""

    @abstractmethod
    def generate(
        self,
        param_space: ParameterSpace,
        n_samples: Optional[int] = None,
        **kwargs
    ) -> List[Treatment]:
        """
        Generate treatments from parameter space.

        Args:
            param_space: Parameter space definition
            n_samples: Number of samples to generate (if applicable)
            **kwargs: Strategy-specific parameters

        Returns:
            List of treatments
        """
        pass


class FullFactorialStrategy(GenerationStrategy):
    """
    Full factorial design - all combinations of parameter values.

    Warning: Can produce very large numbers of treatments!
    """

    def generate(
        self,
        param_space: ParameterSpace,
        n_samples: Optional[int] = None,
        max_treatments: int = 1000,
        **kwargs
    ) -> List[Treatment]:
        """Generate full factorial design."""

        # Get all parameter values
        all_values = []
        param_names = []

        for param_name in sorted(param_space.all_params.keys()):
            param = param_space.all_params[param_name]
            param_names.append(param_name)

            if param.values:
                all_values.append(param.values)
            else:
                # For continuous params, use min/max as endpoints
                all_values.append([param.min_value, param.max_value])

        # Generate all combinations
        combinations = list(itertools.product(*all_values))

        # Check if too many
        if len(combinations) > max_treatments:
            raise ValueError(
                f"Full factorial would produce {len(combinations)} treatments "
                f"(max={max_treatments}). Consider using fractional factorial "
                f"or a sampling strategy."
            )

        # Create treatments
        treatments = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            treatment = Treatment(
                params=params,
                metadata={'strategy': 'full_factorial'}
            )
            treatments.append(treatment)

        return treatments


class FractionalFactorialStrategy(GenerationStrategy):
    """
    Fractional factorial design - subset of full factorial.

    Useful for screening experiments to identify important parameters.
    """

    def generate(
        self,
        param_space: ParameterSpace,
        n_samples: Optional[int] = None,
        fraction: float = 0.5,
        **kwargs
    ) -> List[Treatment]:
        """
        Generate fractional factorial design.

        Args:
            fraction: Fraction of full factorial to sample (0.0 to 1.0)
        """
        # Generate full factorial first
        full_factorial = FullFactorialStrategy().generate(
            param_space,
            max_treatments=100000
        )

        # Sample fraction
        n_to_sample = max(1, int(len(full_factorial) * fraction))
        if n_samples:
            n_to_sample = min(n_to_sample, n_samples)

        sampled = random.sample(full_factorial, n_to_sample)

        # Update metadata
        for treatment in sampled:
            treatment.metadata['strategy'] = 'fractional_factorial'
            treatment.metadata['fraction'] = fraction

        return sampled


class GridSearchStrategy(GenerationStrategy):
    """
    Grid search - uniform grid over continuous parameters.

    For continuous parameters, creates a uniform grid.
    For categorical/ordinal, uses all values.
    """

    def generate(
        self,
        param_space: ParameterSpace,
        n_samples: Optional[int] = None,
        n_points_per_dim: int = 5,
        **kwargs
    ) -> List[Treatment]:
        """
        Generate grid search design.

        Args:
            n_points_per_dim: Number of points per dimension for continuous params
        """
        all_values = []
        param_names = []

        for param_name in sorted(param_space.all_params.keys()):
            param = param_space.all_params[param_name]
            param_names.append(param_name)

            if param.type == ParameterType.CATEGORICAL:
                all_values.append(param.values)

            elif param.type == ParameterType.ORDINAL:
                if param.values:
                    all_values.append(param.values)
                else:
                    # Generate grid
                    grid = np.linspace(
                        param.min_value,
                        param.max_value,
                        n_points_per_dim
                    )
                    all_values.append([int(round(x)) for x in grid])

            else:  # CONTINUOUS
                if param.values:
                    all_values.append(param.values)
                else:
                    grid = np.linspace(
                        param.min_value,
                        param.max_value,
                        n_points_per_dim
                    )
                    all_values.append(list(grid))

        # Generate all combinations
        combinations = list(itertools.product(*all_values))

        # Sample if too many
        if n_samples and len(combinations) > n_samples:
            combinations = random.sample(combinations, n_samples)

        # Create treatments
        treatments = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            treatment = Treatment(
                params=params,
                metadata={'strategy': 'grid_search', 'n_points': n_points_per_dim}
            )
            treatments.append(treatment)

        return treatments


class RandomSamplingStrategy(GenerationStrategy):
    """Random sampling from parameter space."""

    def generate(
        self,
        param_space: ParameterSpace,
        n_samples: int = 20,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Treatment]:
        """Generate random samples."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        treatments = []

        for i in range(n_samples):
            params = {}

            for param_name, param in param_space.all_params.items():
                if param.type == ParameterType.CATEGORICAL:
                    params[param_name] = random.choice(param.values)

                elif param.type == ParameterType.ORDINAL:
                    if param.values:
                        params[param_name] = random.choice(param.values)
                    else:
                        params[param_name] = random.randint(
                            param.min_value,
                            param.max_value
                        )

                else:  # CONTINUOUS
                    if param.values:
                        params[param_name] = random.choice(param.values)
                    else:
                        params[param_name] = random.uniform(
                            param.min_value,
                            param.max_value
                        )

            treatment = Treatment(
                params=params,
                metadata={'strategy': 'random', 'seed': seed, 'index': i}
            )
            treatments.append(treatment)

        return treatments


class LatinHypercubeSamplingStrategy(GenerationStrategy):
    """
    Latin Hypercube Sampling (LHS) - space-filling design.

    Ensures good coverage of parameter space with fewer samples.
    Recommended for most experiments.
    """

    def generate(
        self,
        param_space: ParameterSpace,
        n_samples: int = 20,
        seed: Optional[int] = None,
        criterion: str = 'maximin',
        **kwargs
    ) -> List[Treatment]:
        """
        Generate LHS design.

        Args:
            n_samples: Number of samples
            seed: Random seed for reproducibility
            criterion: Optimization criterion ('maximin', 'correlation', 'ratio')
        """
        # Separate categorical and numerical parameters
        categorical_params = [p for p in param_space.all_params.values()
                            if p.type == ParameterType.CATEGORICAL]
        numerical_params = [p for p in param_space.all_params.values()
                          if p.type in [ParameterType.ORDINAL, ParameterType.CONTINUOUS]]

        treatments = []

        if not numerical_params:
            # Only categorical - use random sampling
            return RandomSamplingStrategy().generate(
                param_space, n_samples, seed
            )

        # Generate LHS for numerical parameters
        sampler = qmc.LatinHypercube(
            d=len(numerical_params),
            seed=seed,
            optimization=criterion
        )
        lhs_samples = sampler.random(n=n_samples)

        # Scale to parameter bounds
        lower_bounds = [p.min_value for p in numerical_params]
        upper_bounds = [p.max_value for p in numerical_params]
        scaled_samples = qmc.scale(lhs_samples, lower_bounds, upper_bounds)

        # Create treatments
        for i, sample in enumerate(scaled_samples):
            params = {}

            # Add numerical parameters
            for j, param in enumerate(numerical_params):
                value = sample[j]

                if param.type == ParameterType.ORDINAL:
                    # Round to nearest valid value
                    if param.values:
                        value = min(param.values, key=lambda x: abs(x - value))
                    else:
                        value = int(round(value))

                params[param.name] = value

            # Add categorical parameters (random)
            if seed is not None:
                random.seed(seed + i)

            for param in categorical_params:
                params[param.name] = random.choice(param.values)

            treatment = Treatment(
                params=params,
                metadata={
                    'strategy': 'lhs',
                    'seed': seed,
                    'criterion': criterion,
                    'index': i
                }
            )
            treatments.append(treatment)

        return treatments


class SobolSequenceStrategy(GenerationStrategy):
    """
    Sobol sequence sampling - low-discrepancy quasi-random sequence.

    Better space-filling properties than random sampling.
    Good for deterministic, reproducible experiments.
    """

    def generate(
        self,
        param_space: ParameterSpace,
        n_samples: int = 20,
        seed: Optional[int] = None,
        scramble: bool = True,
        **kwargs
    ) -> List[Treatment]:
        """
        Generate Sobol sequence design.

        Args:
            n_samples: Number of samples
            seed: Random seed (for scrambling)
            scramble: Whether to scramble the sequence (recommended)
        """
        # Separate categorical and numerical parameters
        categorical_params = [p for p in param_space.all_params.values()
                            if p.type == ParameterType.CATEGORICAL]
        numerical_params = [p for p in param_space.all_params.values()
                          if p.type in [ParameterType.ORDINAL, ParameterType.CONTINUOUS]]

        if not numerical_params:
            return RandomSamplingStrategy().generate(
                param_space, n_samples, seed
            )

        # Generate Sobol sequence
        sampler = qmc.Sobol(
            d=len(numerical_params),
            scramble=scramble,
            seed=seed
        )
        sobol_samples = sampler.random(n=n_samples)

        # Scale to parameter bounds
        lower_bounds = [p.min_value for p in numerical_params]
        upper_bounds = [p.max_value for p in numerical_params]
        scaled_samples = qmc.scale(sobol_samples, lower_bounds, upper_bounds)

        # Create treatments
        treatments = []
        for i, sample in enumerate(scaled_samples):
            params = {}

            # Add numerical parameters
            for j, param in enumerate(numerical_params):
                value = sample[j]

                if param.type == ParameterType.ORDINAL:
                    if param.values:
                        value = min(param.values, key=lambda x: abs(x - value))
                    else:
                        value = int(round(value))

                params[param.name] = value

            # Add categorical parameters
            if seed is not None:
                random.seed(seed + i)

            for param in categorical_params:
                params[param.name] = random.choice(param.values)

            treatment = Treatment(
                params=params,
                metadata={
                    'strategy': 'sobol',
                    'seed': seed,
                    'scramble': scramble,
                    'index': i
                }
            )
            treatments.append(treatment)

        return treatments


# Registry of available strategies
STRATEGY_REGISTRY = {
    'full_factorial': FullFactorialStrategy,
    'fractional_factorial': FractionalFactorialStrategy,
    'grid': GridSearchStrategy,
    'random': RandomSamplingStrategy,
    'lhs': LatinHypercubeSamplingStrategy,
    'sobol': SobolSequenceStrategy,
}


def get_strategy(name: str) -> GenerationStrategy:
    """Get strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name]()

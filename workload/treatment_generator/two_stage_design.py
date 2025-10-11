"""
Two-Stage Experimental Design for Treatment Generation

Stage 1: Categorical Screening
    - Identifies important categorical factor combinations
    - Uses orthogonal arrays or pairwise testing
    - Efficient for high-dimensional categorical spaces

Stage 2: Continuous Optimization
    - Bayesian optimization with Gaussian Process
    - Latin Hypercube Sampling for space-filling initialization
    - Adaptive refinement based on observed performance

This approach is more statistically rigorous than direct LHS on the full space.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from scipy.stats import qmc
from itertools import combinations, product
import logging

from .parameter_space import ParameterSpace, Parameter, ParameterType
from .treatment import Treatment

logger = logging.getLogger(__name__)


class CategoricalScreening:
    """
    Stage 1: Screen categorical factors using orthogonal or pairwise designs.

    Objective: Identify important categorical combinations before exploring
    continuous parameter space.
    """

    def __init__(self, param_space: ParameterSpace):
        """
        Initialize categorical screening.

        Args:
            param_space: Full parameter space
        """
        self.param_space = param_space
        self.categorical_params = [
            p for p in param_space.all_params.values()
            if p.type == ParameterType.CATEGORICAL
        ]
        self.numerical_params = [
            p for p in param_space.all_params.values()
            if p.type in [ParameterType.ORDINAL, ParameterType.CONTINUOUS]
        ]

    def generate_orthogonal_array(
        self,
        strength: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Generate orthogonal array for categorical factors.

        An orthogonal array OA(N, k, v, t) ensures all t-way interactions
        are covered with balanced coverage.

        Args:
            strength: Interaction strength (2 = pairwise, 3 = three-way, etc.)

        Returns:
            List of categorical factor combinations
        """
        if not self.categorical_params:
            return [{}]  # No categorical parameters

        if strength == 2:
            return self._generate_pairwise_covering()
        elif strength == 3:
            return self._generate_three_way_covering()
        else:
            # For higher strength, use full factorial on categoricals
            return self._generate_full_factorial_categorical()

    def _generate_pairwise_covering(self) -> List[Dict[str, Any]]:
        """
        Generate pairwise covering array (all pairs covered).

        Uses IPOG (In-Parameter-Order-General) algorithm or greedy covering.
        """
        if not self.categorical_params:
            return [{}]

        # For simplicity, use AETG-like greedy algorithm
        # In production, could use pyDOE2 or dedicated covering array library

        combinations_to_cover = self._enumerate_pairwise_combinations()

        covering_array = []
        covered = set()

        while len(covered) < len(combinations_to_cover):
            # Generate candidate that covers most uncovered pairs
            candidate = self._generate_max_coverage_candidate(
                combinations_to_cover,
                covered
            )

            covering_array.append(candidate)

            # Mark covered pairs
            for pair_key in self._get_covered_pairs(candidate):
                covered.add(pair_key)

        logger.info(
            f"Pairwise covering: {len(covering_array)} tests cover "
            f"{len(combinations_to_cover)} pairs"
        )

        return covering_array

    def _enumerate_pairwise_combinations(self) -> Dict[str, Any]:
        """
        Enumerate all pairwise combinations that need to be covered.

        Returns:
            Dictionary mapping pair_key → (param1, value1, param2, value2)
        """
        combinations_dict = {}

        # For each pair of categorical parameters
        for p1, p2 in combinations(self.categorical_params, 2):
            # For each value combination
            for v1 in p1.values:
                for v2 in p2.values:
                    key = self._make_pair_key(p1.name, v1, p2.name, v2)
                    combinations_dict[key] = (p1.name, v1, p2.name, v2)

        return combinations_dict

    def _make_pair_key(self, p1: str, v1: Any, p2: str, v2: Any) -> str:
        """Create unique key for parameter-value pair."""
        # Sort to ensure (p1,v1,p2,v2) == (p2,v2,p1,v1)
        if p1 < p2:
            return f"{p1}={v1}|{p2}={v2}"
        else:
            return f"{p2}={v2}|{p1}={v1}"

    def _generate_max_coverage_candidate(
        self,
        all_pairs: Dict[str, Any],
        covered: set
    ) -> Dict[str, Any]:
        """
        Generate candidate that covers maximum uncovered pairs.

        Greedy heuristic: try random candidates, pick best.
        """
        best_candidate = None
        best_coverage = -1

        # Try multiple random candidates
        for _ in range(100):
            candidate = {
                p.name: np.random.choice(p.values)
                for p in self.categorical_params
            }

            # Count newly covered pairs
            newly_covered = 0
            for pair_key in self._get_covered_pairs(candidate):
                if pair_key not in covered and pair_key in all_pairs:
                    newly_covered += 1

            if newly_covered > best_coverage:
                best_coverage = newly_covered
                best_candidate = candidate

        return best_candidate

    def _get_covered_pairs(self, candidate: Dict[str, Any]) -> List[str]:
        """Get all pairs covered by this candidate."""
        covered_pairs = []

        param_names = [p.name for p in self.categorical_params]

        for p1, p2 in combinations(param_names, 2):
            if p1 in candidate and p2 in candidate:
                key = self._make_pair_key(
                    p1, candidate[p1],
                    p2, candidate[p2]
                )
                covered_pairs.append(key)

        return covered_pairs

    def _generate_three_way_covering(self) -> List[Dict[str, Any]]:
        """
        Generate three-way covering array.

        More expensive but covers all 3-way interactions.
        """
        # For 3-way covering, use similar greedy approach
        # but with 3-tuples instead of pairs

        # Simplified: use full factorial for now
        # In production, implement proper 3-way covering algorithm
        logger.warning("Three-way covering not fully implemented, using full factorial")
        return self._generate_full_factorial_categorical()

    def _generate_full_factorial_categorical(self) -> List[Dict[str, Any]]:
        """Generate full factorial on categorical parameters."""
        if not self.categorical_params:
            return [{}]

        param_values = [p.values for p in self.categorical_params]
        param_names = [p.name for p in self.categorical_params]

        full_factorial = []
        for combo in product(*param_values):
            full_factorial.append(dict(zip(param_names, combo)))

        return full_factorial

    def generate_fractional_factorial(
        self,
        resolution: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Generate fractional factorial design for categorical factors.

        Args:
            resolution: Design resolution (3, 4, or 5)
                - Resolution III: Main effects not confounded
                - Resolution IV: Main effects clear, some 2-way interactions confounded
                - Resolution V: Main + 2-way effects clear

        Returns:
            List of categorical factor combinations
        """
        if not self.categorical_params:
            return [{}]

        # For 2-level factors, use standard fractional factorial
        # For multi-level, use orthogonal Latin hypercube

        # Check if all factors are 2-level
        all_two_level = all(len(p.values) == 2 for p in self.categorical_params)

        if all_two_level:
            return self._generate_2level_fractional_factorial(resolution)
        else:
            # For mixed levels, use orthogonal array
            return self.generate_orthogonal_array(strength=2)

    def _generate_2level_fractional_factorial(
        self,
        resolution: int
    ) -> List[Dict[str, Any]]:
        """
        Generate 2^(k-p) fractional factorial design.

        For k factors, generates 2^(k-p) runs where p is chosen based on resolution.
        """
        k = len(self.categorical_params)

        # Determine number of runs based on resolution
        if resolution == 3:
            # Resolution III: 2^(k-1) runs
            p = k - int(np.ceil(np.log2(k)))
        elif resolution == 4:
            # Resolution IV: More runs
            p = max(0, k - int(np.ceil(np.log2(k))) - 1)
        else:
            # Resolution V or higher: Even more runs
            p = max(0, k - int(np.ceil(np.log2(k))) - 2)

        n_runs = 2 ** (k - p)

        # Generate base design
        base_factors = k - p
        base_design = self._generate_full_factorial_2level(base_factors)

        # Add derived factors based on generators
        # Simplified: just use base design for now
        # In production, implement proper confounding structure

        # Map -1/+1 to actual categorical values
        treatments = []
        for row in base_design[:n_runs]:
            treatment = {}
            for i, param in enumerate(self.categorical_params[:base_factors]):
                # Map -1 → first value, +1 → second value
                value_idx = 0 if row[i] < 0 else 1
                treatment[param.name] = param.values[value_idx]

            treatments.append(treatment)

        logger.info(
            f"Fractional factorial: 2^({k}-{p}) = {len(treatments)} runs "
            f"(Resolution {resolution})"
        )

        return treatments

    def _generate_full_factorial_2level(self, k: int) -> np.ndarray:
        """Generate full 2^k factorial design with -1/+1 coding."""
        n_runs = 2 ** k
        design = np.zeros((n_runs, k))

        for i in range(k):
            # Create alternating pattern
            period = 2 ** (k - i - 1)
            design[:, i] = np.tile(
                np.repeat([-1, 1], period),
                2 ** i
            )

        return design


class BayesianOptimizationStage:
    """
    Stage 2: Bayesian optimization for numerical parameters.

    Given categorical combinations from Stage 1, optimize numerical parameters
    using Gaussian Process with LHS initialization.
    """

    def __init__(
        self,
        param_space: ParameterSpace,
        categorical_config: Dict[str, Any]
    ):
        """
        Initialize Bayesian optimization stage.

        Args:
            param_space: Full parameter space
            categorical_config: Fixed categorical parameters from Stage 1
        """
        self.param_space = param_space
        self.categorical_config = categorical_config

        # Filter to numerical parameters only
        self.numerical_params = [
            p for p in param_space.all_params.values()
            if p.type in [ParameterType.ORDINAL, ParameterType.CONTINUOUS]
        ]

        # Initialize GP (lazily, when we have data)
        self.gp = None
        self.X_observed = []
        self.y_observed = []

    def generate_initial_lhs(
        self,
        n_samples: int,
        criterion: str = 'maximin'
    ) -> List[Treatment]:
        """
        Generate initial space-filling design using LHS.

        Args:
            n_samples: Number of initial samples
            criterion: LHS optimization criterion

        Returns:
            List of treatments with categorical config + LHS numerical values
        """
        if not self.numerical_params:
            # No numerical parameters, return single treatment
            return [Treatment(
                params=self.categorical_config.copy(),
                metadata={'stage': 'initial_lhs', 'index': 0}
            )]

        # Generate LHS for numerical parameters
        sampler = qmc.LatinHypercube(
            d=len(self.numerical_params),
            optimization=criterion
        )
        lhs_samples = sampler.random(n=n_samples)

        # Scale to parameter bounds
        lower_bounds = [p.min_value for p in self.numerical_params]
        upper_bounds = [p.max_value for p in self.numerical_params]
        scaled_samples = qmc.scale(lhs_samples, lower_bounds, upper_bounds)

        # Create treatments
        treatments = []
        for i, sample in enumerate(scaled_samples):
            params = self.categorical_config.copy()

            for j, param in enumerate(self.numerical_params):
                value = sample[j]

                if param.type == ParameterType.ORDINAL:
                    # Round to nearest valid value
                    if param.values:
                        value = min(param.values, key=lambda x: abs(x - value))
                    else:
                        value = int(round(value))

                params[param.name] = value

            treatment = Treatment(
                params=params,
                metadata={
                    'stage': 'initial_lhs',
                    'categorical_config': self.categorical_config,
                    'index': i
                }
            )
            treatments.append(treatment)

        return treatments

    def update_with_observations(
        self,
        treatments: List[Treatment],
        objectives: List[float]
    ):
        """
        Update GP model with new observations.

        Args:
            treatments: List of evaluated treatments
            objectives: Corresponding objective values (e.g., throughput)
        """
        # Encode treatments (numerical parameters only)
        for treatment, objective in zip(treatments, objectives):
            # Extract numerical parameters
            numerical_values = []
            for param in self.numerical_params:
                value = treatment.params.get(param.name)
                if value is not None:
                    normalized = param.normalize(value)
                    numerical_values.append(normalized)

            self.X_observed.append(numerical_values)
            self.y_observed.append(objective)

        # Fit GP
        if len(self.X_observed) >= 3:
            self._fit_gp()

    def _fit_gp(self):
        """Fit Gaussian Process on observed data."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel

            X = np.array(self.X_observed)
            y = np.array(self.y_observed)

            # Define kernel
            kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0)

            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )

            self.gp.fit(X, y)

            logger.info(
                f"GP fitted with {len(X)} observations, "
                f"kernel: {self.gp.kernel_}"
            )

        except ImportError:
            logger.warning("scikit-learn not available, skipping GP fitting")
            self.gp = None

    def suggest_next(
        self,
        acquisition: str = 'ei',
        n_candidates: int = 1000
    ) -> Treatment:
        """
        Suggest next treatment using acquisition function.

        Args:
            acquisition: Acquisition function ('ei', 'ucb', 'poi')
            n_candidates: Number of candidates to evaluate

        Returns:
            Next treatment to evaluate
        """
        if self.gp is None or len(self.X_observed) < 3:
            # Not enough data, use random sampling
            logger.info("Insufficient data for GP, using random sampling")
            return self.generate_initial_lhs(n_samples=1)[0]

        # Generate candidate treatments
        candidates = self.generate_initial_lhs(n_samples=n_candidates)

        best_treatment = None
        best_acq_value = -np.inf

        for candidate in candidates:
            # Encode candidate
            x_cand = []
            for param in self.numerical_params:
                value = candidate.params.get(param.name)
                if value is not None:
                    x_cand.append(param.normalize(value))

            x_cand = np.array(x_cand).reshape(1, -1)

            # Compute acquisition function
            if acquisition == 'ei':
                acq_value = self._expected_improvement(x_cand)
            elif acquisition == 'ucb':
                acq_value = self._upper_confidence_bound(x_cand)
            elif acquisition == 'poi':
                acq_value = self._probability_of_improvement(x_cand)
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition}")

            if acq_value > best_acq_value:
                best_acq_value = acq_value
                best_treatment = candidate

        logger.info(
            f"Selected treatment with {acquisition}={best_acq_value:.4f}"
        )

        return best_treatment

    def _expected_improvement(self, x: np.ndarray) -> float:
        """Compute Expected Improvement acquisition function."""
        mu, sigma = self.gp.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]

        if sigma == 0:
            return 0

        from scipy.stats import norm
        y_best = np.max(self.y_observed)
        z = (mu - y_best) / sigma

        ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def _upper_confidence_bound(self, x: np.ndarray, beta: float = 2.0) -> float:
        """Compute Upper Confidence Bound acquisition function."""
        mu, sigma = self.gp.predict(x, return_std=True)
        return mu[0] + beta * sigma[0]

    def _probability_of_improvement(self, x: np.ndarray) -> float:
        """Compute Probability of Improvement acquisition function."""
        mu, sigma = self.gp.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]

        if sigma == 0:
            return 0

        from scipy.stats import norm
        y_best = np.max(self.y_observed)
        z = (mu - y_best) / sigma

        return norm.cdf(z)


class TwoStageTreatmentGenerator:
    """
    Two-stage treatment generator combining categorical screening
    with Bayesian optimization.

    Usage:
        generator = TwoStageTreatmentGenerator(param_space)

        # Stage 1: Generate categorical configurations
        categorical_configs = generator.stage1_screening(method='pairwise')

        # Stage 2: For each categorical config, generate numerical treatments
        for config in categorical_configs:
            # Initial LHS
            treatments = generator.stage2_initial(config, n_samples=10)

            # Evaluate treatments...
            # results = evaluate(treatments)

            # Bayesian optimization
            generator.stage2_update(config, treatments, results)
            next_treatment = generator.stage2_suggest(config, acquisition='ei')
    """

    def __init__(self, param_space: ParameterSpace):
        """
        Initialize two-stage generator.

        Args:
            param_space: Full parameter space
        """
        self.param_space = param_space
        self.screening = CategoricalScreening(param_space)

        # Track Bayesian optimization state for each categorical config
        self.bo_stages: Dict[str, BayesianOptimizationStage] = {}

    def stage1_screening(
        self,
        method: str = 'pairwise',
        strength: int = 2,
        resolution: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Generate categorical factor combinations.

        Args:
            method: Screening method
                - 'pairwise': Pairwise covering array
                - 'orthogonal': Orthogonal array (strength-way covering)
                - 'fractional': Fractional factorial (for 2-level factors)
                - 'full': Full factorial on categoricals
            strength: Interaction strength for orthogonal arrays (2, 3, ...)
            resolution: Resolution for fractional factorial (3, 4, 5)

        Returns:
            List of categorical factor configurations
        """
        logger.info(f"Stage 1: Categorical screening with method='{method}'")

        if method == 'pairwise':
            configs = self.screening.generate_orthogonal_array(strength=2)

        elif method == 'orthogonal':
            configs = self.screening.generate_orthogonal_array(strength=strength)

        elif method == 'fractional':
            configs = self.screening.generate_fractional_factorial(resolution=resolution)

        elif method == 'full':
            configs = self.screening._generate_full_factorial_categorical()

        else:
            raise ValueError(
                f"Unknown screening method: {method}. "
                f"Choose from: pairwise, orthogonal, fractional, full"
            )

        logger.info(f"Generated {len(configs)} categorical configurations")

        return configs

    def stage2_initial(
        self,
        categorical_config: Dict[str, Any],
        n_samples: int = 10,
        criterion: str = 'maximin'
    ) -> List[Treatment]:
        """
        Stage 2: Generate initial LHS samples for given categorical config.

        Args:
            categorical_config: Fixed categorical parameters
            n_samples: Number of LHS samples
            criterion: LHS optimization criterion

        Returns:
            List of treatments with LHS numerical parameters
        """
        # Create or retrieve BO stage
        config_key = self._config_to_key(categorical_config)

        if config_key not in self.bo_stages:
            self.bo_stages[config_key] = BayesianOptimizationStage(
                self.param_space,
                categorical_config
            )

        bo_stage = self.bo_stages[config_key]

        logger.info(
            f"Stage 2: Initial LHS for config {config_key}, n_samples={n_samples}"
        )

        return bo_stage.generate_initial_lhs(n_samples, criterion)

    def stage2_update(
        self,
        categorical_config: Dict[str, Any],
        treatments: List[Treatment],
        objectives: List[float]
    ):
        """
        Update Bayesian optimization model with observations.

        Args:
            categorical_config: Categorical configuration
            treatments: Evaluated treatments
            objectives: Objective values (e.g., throughput)
        """
        config_key = self._config_to_key(categorical_config)

        if config_key not in self.bo_stages:
            raise ValueError(f"No BO stage for config {config_key}")

        bo_stage = self.bo_stages[config_key]
        bo_stage.update_with_observations(treatments, objectives)

        logger.info(
            f"Updated BO model for config {config_key} "
            f"with {len(treatments)} observations"
        )

    def stage2_suggest(
        self,
        categorical_config: Dict[str, Any],
        acquisition: str = 'ei',
        n_candidates: int = 1000
    ) -> Treatment:
        """
        Suggest next treatment using Bayesian optimization.

        Args:
            categorical_config: Categorical configuration
            acquisition: Acquisition function ('ei', 'ucb', 'poi')
            n_candidates: Number of candidates to evaluate

        Returns:
            Next treatment to evaluate
        """
        config_key = self._config_to_key(categorical_config)

        if config_key not in self.bo_stages:
            raise ValueError(f"No BO stage for config {config_key}")

        bo_stage = self.bo_stages[config_key]

        return bo_stage.suggest_next(acquisition, n_candidates)

    def _config_to_key(self, config: Dict[str, Any]) -> str:
        """Convert categorical config to unique key."""
        sorted_items = sorted(config.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)

    def generate_full_design(
        self,
        stage1_method: str = 'pairwise',
        stage2_samples_per_config: int = 10,
        stage2_criterion: str = 'maximin'
    ) -> List[Treatment]:
        """
        Generate complete two-stage design.

        Convenience method that runs both stages and returns all treatments.

        Args:
            stage1_method: Categorical screening method
            stage2_samples_per_config: LHS samples per categorical config
            stage2_criterion: LHS optimization criterion

        Returns:
            Complete list of treatments from both stages
        """
        # Stage 1: Categorical screening
        categorical_configs = self.stage1_screening(method=stage1_method)

        # Stage 2: LHS for each config
        all_treatments = []

        for i, config in enumerate(categorical_configs):
            logger.info(
                f"Generating treatments for categorical config {i+1}/{len(categorical_configs)}"
            )

            treatments = self.stage2_initial(
                config,
                n_samples=stage2_samples_per_config,
                criterion=stage2_criterion
            )

            all_treatments.extend(treatments)

        logger.info(
            f"Total treatments generated: {len(all_treatments)} "
            f"({len(categorical_configs)} configs × {stage2_samples_per_config} samples)"
        )

        return all_treatments

"""
Parameter space definition and encoding for treatment generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np


def _parse_level_value(value: Any) -> Any:
    """Best-effort conversion of legacy level values to numbers."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped == "":
        return stripped

    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


class ParameterType(Enum):
    """Parameter types for different sampling strategies."""
    CATEGORICAL = "categorical"  # Discrete, unordered (e.g., "none", "prefix", "kv")
    ORDINAL = "ordinal"          # Discrete, ordered (e.g., 1, 2, 4, 8)
    CONTINUOUS = "continuous"    # Continuous values (e.g., 0.7, 0.8, 0.9)


@dataclass
class Parameter:
    """
    Definition of a single parameter in the experiment space.

    Attributes:
        name: Parameter name (e.g., "LLMDBENCH_VLLM_COMMON_REPLICAS")
        type: Parameter type (categorical, ordinal, continuous)
        values: Explicit list of possible values
        min_value: Minimum value (for ordinal/continuous)
        max_value: Maximum value (for ordinal/continuous)
        default: Default value if not specified
        description: Human-readable description
        unit: Unit of measurement (e.g., "tokens", "GB", "ms")
    """
    name: str
    type: ParameterType
    values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    default: Optional[Any] = None
    description: str = ""
    unit: str = ""

    def __post_init__(self):
        """Validate parameter definition."""
        if self.type == ParameterType.CATEGORICAL:
            if not self.values:
                raise ValueError(f"Categorical parameter '{self.name}' must have values")

        if self.type in [ParameterType.ORDINAL, ParameterType.CONTINUOUS]:
            if self.values:
                # Infer min/max from values if not provided
                if self.min_value is None:
                    self.min_value = min(self.values)
                if self.max_value is None:
                    self.max_value = max(self.values)
            elif self.min_value is None or self.max_value is None:
                raise ValueError(
                    f"Parameter '{self.name}' must have either values or min/max_value"
                )

    @property
    def domain(self) -> tuple:
        """Get the domain (min, max) for numerical parameters."""
        if self.type == ParameterType.CATEGORICAL:
            return None
        return (self.min_value, self.max_value)

    def is_valid(self, value: Any) -> bool:
        """Check if a value is valid for this parameter."""
        if self.values and value not in self.values:
            return False

        if self.type in [ParameterType.ORDINAL, ParameterType.CONTINUOUS]:
            if value < self.min_value or value > self.max_value:
                return False

        return True

    def normalize(self, value: Any) -> float:
        """
        Normalize value to [0, 1] for encoding.

        Used for distance calculations and GP encoding.
        """
        if self.type == ParameterType.CATEGORICAL:
            # One-hot encoding handled separately
            raise ValueError("Use one-hot encoding for categorical parameters")

        if self.max_value == self.min_value:
            return 0.5

        return (value - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, normalized_value: float) -> Any:
        """
        Convert normalized [0, 1] value back to parameter space.
        """
        if self.type == ParameterType.CATEGORICAL:
            raise ValueError("Categorical parameters cannot be denormalized")

        value = self.min_value + normalized_value * (self.max_value - self.min_value)

        if self.type == ParameterType.ORDINAL:
            # Round to nearest valid value
            if self.values:
                return min(self.values, key=lambda x: abs(x - value))
            else:
                return int(round(value))

        return value

    @classmethod
    def from_dict(cls, name: str, config: Dict) -> "Parameter":
        """
        Create Parameter from dictionary configuration.

        Examples:
            {'type': 'categorical', 'values': ['none', 'prefix', 'kv']}
            {'type': 'ordinal', 'values': [1, 2, 4, 8]}
            {'type': 'continuous', 'min': 0.7, 'max': 0.95}
        """
        param_type = ParameterType(config.get('type', 'categorical'))

        return cls(
            name=name,
            type=param_type,
            values=config.get('values'),
            min_value=config.get('min', config.get('min_value')),
            max_value=config.get('max', config.get('max_value')),
            default=config.get('default'),
            description=config.get('description', ''),
            unit=config.get('unit', '')
        )

    @classmethod
    def infer_from_values(cls, name: str, values: List[Any]) -> "Parameter":
        """
        Infer parameter type from values.

        Heuristics:
        - All strings → categorical
        - All integers → ordinal
        - All floats → continuous
        """
        if not values:
            raise ValueError(f"Cannot infer type for parameter '{name}' with empty values")

        # Check if all values are strings
        if all(isinstance(v, str) for v in values):
            return cls(name=name, type=ParameterType.CATEGORICAL, values=values)

        # Check if all values are integers
        if all(isinstance(v, (int, np.integer)) for v in values):
            return cls(
                name=name,
                type=ParameterType.ORDINAL,
                values=sorted(values),
                min_value=min(values),
                max_value=max(values)
            )

        # Otherwise, treat as continuous
        return cls(
            name=name,
            type=ParameterType.CONTINUOUS,
            values=sorted(values),
            min_value=min(values),
            max_value=max(values)
        )


class ParameterSpace:
    """
    Complete parameter space definition for an experiment.

    Manages both setup (infrastructure) and run (workload) parameters.
    """

    def __init__(self):
        self.setup_params: Dict[str, Parameter] = {}
        self.run_params: Dict[str, Parameter] = {}

    def add_parameter(
        self,
        param: Parameter,
        phase: str = "setup"
    ):
        """Add a parameter to the space."""
        if phase == "setup":
            self.setup_params[param.name] = param
        elif phase == "run":
            self.run_params[param.name] = param
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 'setup' or 'run'")

    @property
    def all_params(self) -> Dict[str, Parameter]:
        """Get all parameters (setup + run)."""
        return {**self.setup_params, **self.run_params}

    @property
    def categorical_params(self) -> List[Parameter]:
        """Get all categorical parameters."""
        return [p for p in self.all_params.values()
                if p.type == ParameterType.CATEGORICAL]

    @property
    def numerical_params(self) -> List[Parameter]:
        """Get all numerical (ordinal + continuous) parameters."""
        return [p for p in self.all_params.values()
                if p.type in [ParameterType.ORDINAL, ParameterType.CONTINUOUS]]

    def encode_treatment(self, treatment: Dict[str, Any]) -> np.ndarray:
        """
        Encode treatment to numerical vector for GP/optimization.

        Uses one-hot encoding for categorical, normalization for numerical.

        Returns:
            1D numpy array of encoded values
        """
        encoded = []

        # Process in consistent order
        for param_name in sorted(self.all_params.keys()):
            param = self.all_params[param_name]
            value = treatment.get(param_name)

            if value is None:
                continue

            if param.type == ParameterType.CATEGORICAL:
                # One-hot encoding
                one_hot = [0] * len(param.values)
                if value in param.values:
                    idx = param.values.index(value)
                    one_hot[idx] = 1
                encoded.extend(one_hot)
            else:
                # Normalize to [0, 1]
                normalized = param.normalize(value)
                encoded.append(normalized)

        return np.array(encoded)

    def decode_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Decode numerical vector back to treatment.

        Inverse of encode_treatment.
        """
        treatment = {}
        idx = 0

        for param_name in sorted(self.all_params.keys()):
            param = self.all_params[param_name]

            if param.type == ParameterType.CATEGORICAL:
                # Decode one-hot
                n_values = len(param.values)
                one_hot = vector[idx:idx + n_values]
                category_idx = np.argmax(one_hot)
                treatment[param_name] = param.values[category_idx]
                idx += n_values
            else:
                # Denormalize
                normalized_value = vector[idx]
                treatment[param_name] = param.denormalize(normalized_value)
                idx += 1

        return treatment

    @property
    def encoded_dimension(self) -> int:
        """Get the dimension of encoded vectors."""
        dim = 0
        for param in self.all_params.values():
            if param.type == ParameterType.CATEGORICAL:
                dim += len(param.values)
            else:
                dim += 1
        return dim

    @classmethod
    def from_yaml_dict(cls, yaml_dict: Dict) -> "ParameterSpace":
        """
        Create ParameterSpace from experiment YAML dictionary.

        Supports both old format (factors + levels) and new format (parameters).
        """
        space = cls()

        # Parse setup parameters
        if 'setup' in yaml_dict:
            setup = yaml_dict['setup']

            # Old format: factors + levels
            if 'factors' in setup and 'levels' in setup:
                for factor in setup['factors']:
                    level_str = setup['levels'].get(factor, '')
                    values = [
                        _parse_level_value(v)
                        for v in level_str.split(',')
                        if v.strip() != ''
                    ]

                    param = Parameter.infer_from_values(factor, values)
                    space.add_parameter(param, phase='setup')

            # New format: parameters list
            elif 'parameters' in setup:
                for param_config in setup['parameters']:
                    name = param_config['name']
                    param = Parameter.from_dict(name, param_config)
                    space.add_parameter(param, phase='setup')

        # Parse run parameters (same logic)
        if 'run' in yaml_dict:
            run = yaml_dict['run']

            if 'factors' in run and 'levels' in run:
                for factor in run['factors']:
                    level_str = run['levels'].get(factor, '')
                    values = [
                        _parse_level_value(v)
                        for v in level_str.split(',')
                        if v.strip() != ''
                    ]

                    param = Parameter.infer_from_values(factor, values)
                    space.add_parameter(param, phase='run')

            elif 'parameters' in run:
                for param_config in run['parameters']:
                    name = param_config['name']
                    param = Parameter.from_dict(name, param_config)
                    space.add_parameter(param, phase='run')

        return space

    def __repr__(self) -> str:
        return (f"ParameterSpace(setup={len(self.setup_params)}, "
                f"run={len(self.run_params)}, "
                f"encoded_dim={self.encoded_dimension})")

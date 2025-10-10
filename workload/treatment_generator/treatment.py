"""
Treatment representation and utilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime
import hashlib
import json


@dataclass
class Treatment:
    """
    Represents a single experiment treatment.

    A treatment is a specific combination of parameter values that will be
    tested during the experiment.

    Attributes:
        params: Dictionary of parameter name → value
        setup_params: Setup (infrastructure) parameters only
        run_params: Run (workload) parameters only
        treatment_id: Unique identifier for this treatment
        metadata: Additional metadata (e.g., generation strategy)
        is_feasible: Whether this treatment passed feasibility checks
        feasibility_reason: Reason if infeasible
    """
    params: Dict[str, Any]
    setup_params: Dict[str, Any] = field(default_factory=dict)
    run_params: Dict[str, Any] = field(default_factory=dict)
    treatment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_feasible: bool = True
    feasibility_reason: str = ""

    def __post_init__(self):
        """Initialize treatment ID if not provided."""
        if self.treatment_id is None:
            self.treatment_id = self._generate_id()

        # Split params into setup and run if not already done
        if not self.setup_params and not self.run_params:
            for key, value in self.params.items():
                if key.startswith('LLMDBENCH_'):
                    self.setup_params[key] = value
                else:
                    self.run_params[key] = value

    def _generate_id(self) -> str:
        """Generate deterministic ID from parameters."""
        # Sort params for consistent hashing
        sorted_params = json.dumps(self.params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()[:12]

    def to_csv_string(self) -> str:
        """
        Convert to CSV string format (compatible with existing code).

        Format: "value1,value2,value3,..."
        """
        # Get values in consistent order
        sorted_keys = sorted(self.params.keys())
        values = [str(self.params[k]) for k in sorted_keys]
        return ",".join(values)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'treatment_id': self.treatment_id,
            'params': self.params,
            'setup_params': self.setup_params,
            'run_params': self.run_params,
            'is_feasible': self.is_feasible,
            'feasibility_reason': self.feasibility_reason,
            'metadata': self.metadata
        }

    @classmethod
    def from_csv_string(cls, csv_str: str, param_names: list) -> "Treatment":
        """
        Create Treatment from CSV string.

        Args:
            csv_str: Comma-separated values
            param_names: List of parameter names in order
        """
        values = [v.strip() for v in csv_str.split(',')]

        # Try to convert to numbers
        converted_values = []
        for v in values:
            if v == 'NA':
                converted_values.append(None)
            elif v.isdigit():
                converted_values.append(int(v))
            else:
                try:
                    converted_values.append(float(v))
                except ValueError:
                    converted_values.append(v)

        params = dict(zip(param_names, converted_values))
        return cls(params=params)

    def __repr__(self) -> str:
        return f"Treatment(id={self.treatment_id}, params={len(self.params)})"

    def __eq__(self, other) -> bool:
        """Treatments are equal if they have the same parameters."""
        if not isinstance(other, Treatment):
            return False
        return self.params == other.params

    def __hash__(self) -> int:
        """Hash based on treatment ID."""
        return hash(self.treatment_id)

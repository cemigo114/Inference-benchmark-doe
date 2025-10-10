"""
Treatment Generator for LLM-D Benchmark

A self-contained module for generating valid experiment treatments with:
- Multiple sampling strategies (LHS, Sobol, factorial, random)
- Two-stage design (categorical screening + Bayesian optimization)
- Constraint validation via Capacity Planner
- Support for categorical, ordinal, and continuous parameters
- Bayesian optimization with Gaussian Process
- Backwards compatible with existing experiment YAML format
"""

from .generator import TreatmentGenerator
from .parameter_space import ParameterSpace, Parameter, ParameterType
from .treatment import Treatment
from .validators import CapacityPlannerValidator, ConstraintValidator
from .two_stage_design import (
    TwoStageTreatmentGenerator,
    CategoricalScreening,
    BayesianOptimizationStage
)

__version__ = "0.2.0"

__all__ = [
    "TreatmentGenerator",
    "TwoStageTreatmentGenerator",
    "ParameterSpace",
    "Parameter",
    "ParameterType",
    "Treatment",
    "CapacityPlannerValidator",
    "ConstraintValidator",
    "CategoricalScreening",
    "BayesianOptimizationStage",
]

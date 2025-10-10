"""
Main TreatmentGenerator class - orchestrates treatment generation.
"""

from typing import List, Dict, Any, Optional, Callable
import logging
import yaml

from .parameter_space import ParameterSpace
from .treatment import Treatment
from .strategies import get_strategy, STRATEGY_REGISTRY
from .validators import (
    Validator,
    CompositeValidator,
    ConstraintValidator,
    CapacityPlannerValidator,
    TotalGPUValidator,
    NAValueValidator
)

logger = logging.getLogger(__name__)


class TreatmentGenerator:
    """
    Main class for generating experiment treatments.

    Orchestrates parameter space definition, treatment generation strategies,
    and constraint validation.

    Example:
        >>> generator = TreatmentGenerator.from_yaml_file("experiment.yaml")
        >>> treatments = generator.generate(strategy='lhs', n_samples=20)
        >>> valid_treatments = generator.filter_feasible(treatments)
    """

    def __init__(
        self,
        param_space: Optional[ParameterSpace] = None,
        validators: Optional[List[Validator]] = None
    ):
        """
        Initialize treatment generator.

        Args:
            param_space: Parameter space definition
            validators: List of validators for constraint checking
        """
        self.param_space = param_space or ParameterSpace()
        self.validators = validators or []
        self.validator = CompositeValidator(self.validators)

        # Generation history
        self.generated_treatments: List[Treatment] = []
        self.feasible_treatments: List[Treatment] = []
        self.infeasible_treatments: List[Treatment] = []

    def add_validator(self, validator: Validator):
        """Add a validator."""
        self.validators.append(validator)
        self.validator = CompositeValidator(self.validators)

    def generate(
        self,
        strategy: str = 'lhs',
        n_samples: Optional[int] = 20,
        validate: bool = True,
        **kwargs
    ) -> List[Treatment]:
        """
        Generate treatments using specified strategy.

        Args:
            strategy: Generation strategy name
                ('lhs', 'sobol', 'random', 'factorial', 'grid')
            n_samples: Number of samples to generate
            validate: Whether to validate treatments
            **kwargs: Strategy-specific parameters

        Returns:
            List of treatments (validated if validate=True)
        """
        # Get strategy
        strategy_obj = get_strategy(strategy)

        # Generate treatments
        logger.info(
            f"Generating treatments with strategy='{strategy}', "
            f"n_samples={n_samples}"
        )

        treatments = strategy_obj.generate(
            self.param_space,
            n_samples=n_samples,
            **kwargs
        )

        self.generated_treatments.extend(treatments)

        logger.info(f"Generated {len(treatments)} treatments")

        # Validate if requested
        if validate:
            treatments = self.filter_feasible(treatments)

        return treatments

    def filter_feasible(
        self,
        treatments: List[Treatment],
        mark_infeasible: bool = True
    ) -> List[Treatment]:
        """
        Filter treatments to only feasible ones.

        Args:
            treatments: List of treatments to filter
            mark_infeasible: If True, mark infeasible treatments in-place

        Returns:
            List of feasible treatments
        """
        feasible = []
        infeasible = []

        for treatment in treatments:
            is_valid, reason = self.validator.validate(treatment)

            if is_valid:
                feasible.append(treatment)
                if mark_infeasible:
                    treatment.is_feasible = True
            else:
                infeasible.append(treatment)
                if mark_infeasible:
                    treatment.is_feasible = False
                    treatment.feasibility_reason = reason

                logger.debug(
                    f"Treatment {treatment.treatment_id} infeasible: {reason}"
                )

        logger.info(
            f"Filtered {len(feasible)} feasible treatments "
            f"({len(infeasible)} infeasible)"
        )

        self.feasible_treatments.extend(feasible)
        self.infeasible_treatments.extend(infeasible)

        return feasible

    def generate_from_explicit_list(
        self,
        treatments_list: List[str],
        validate: bool = True
    ) -> List[Treatment]:
        """
        Generate treatments from explicit list (old YAML format).

        Args:
            treatments_list: List of CSV strings (e.g., ["1,2,3", "4,5,6"])
            validate: Whether to validate

        Returns:
            List of treatments
        """
        treatments = []
        param_names = sorted(self.param_space.all_params.keys())

        for csv_str in treatments_list:
            treatment = Treatment.from_csv_string(csv_str, param_names)
            treatment.metadata['strategy'] = 'explicit'
            treatments.append(treatment)

        self.generated_treatments.extend(treatments)

        if validate:
            treatments = self.filter_feasible(treatments)

        return treatments

    @classmethod
    def from_yaml_file(
        cls,
        yaml_path: str,
        auto_add_validators: bool = True
    ) -> "TreatmentGenerator":
        """
        Create generator from YAML file.

        Args:
            yaml_path: Path to experiment YAML file
            auto_add_validators: Automatically add standard validators

        Returns:
            Initialized TreatmentGenerator
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls.from_yaml_dict(config, auto_add_validators)

    @classmethod
    def from_yaml_dict(
        cls,
        yaml_dict: Dict,
        auto_add_validators: bool = True
    ) -> "TreatmentGenerator":
        """
        Create generator from YAML dictionary.

        Args:
            yaml_dict: Experiment configuration dictionary
            auto_add_validators: Automatically add standard validators

        Returns:
            Initialized TreatmentGenerator
        """
        # Parse parameter space
        param_space = ParameterSpace.from_yaml_dict(yaml_dict)

        # Initialize generator
        generator = cls(param_space=param_space)

        # Add validators
        if auto_add_validators:
            generator._add_default_validators(yaml_dict)

        # Add custom constraints if specified
        if 'constraints' in yaml_dict:
            constraints = yaml_dict['constraints']

            if isinstance(constraints, list):
                # List of constraint expressions
                generator.add_validator(ConstraintValidator(constraints))

            elif isinstance(constraints, dict):
                # Dictionary with constraint types
                if 'expressions' in constraints:
                    generator.add_validator(
                        ConstraintValidator(constraints['expressions'])
                    )

        return generator

    def _add_default_validators(self, yaml_dict: Dict):
        """Add default validators based on config."""
        # Add NA value validator
        self.add_validator(NAValueValidator())

        # Add total GPU validator if cluster info available
        if 'cluster' in yaml_dict:
            cluster_config = yaml_dict['cluster']
            if 'total_gpus' in cluster_config:
                self.add_validator(
                    TotalGPUValidator(cluster_config['total_gpus'])
                )

        # Add capacity planner validator if model info available
        model_name = None
        gpu_memory_gb = None
        total_gpus = None

        # Try to extract from various locations in YAML
        if 'model' in yaml_dict:
            model_config = yaml_dict['model']
            model_name = model_config.get('name')

        if 'cluster' in yaml_dict:
            cluster_config = yaml_dict['cluster']
            gpu_memory_gb = cluster_config.get('gpu_memory_gb')
            total_gpus = cluster_config.get('total_gpus')

        if model_name:
            self.add_validator(
                CapacityPlannerValidator(
                    model_name=model_name,
                    gpu_memory_gb=gpu_memory_gb,
                    total_gpus=total_gpus,
                    strict=False  # Non-strict by default
                )
            )

    def to_legacy_yaml(
        self,
        treatments: List[Treatment],
        output_path: str
    ):
        """
        Export treatments to legacy YAML format.

        For backwards compatibility with existing e2e.sh script.
        """
        # Separate setup and run treatments
        setup_treatments = []
        run_treatments = []

        for treatment in treatments:
            # Setup treatments
            if treatment.setup_params:
                csv = treatment.to_csv_string()
                setup_treatments.append(csv)

            # Run treatments
            if treatment.run_params:
                csv = ",".join(str(v) for v in treatment.run_params.values())
                run_treatments.append(csv)

        # Build YAML structure
        yaml_dict = {}

        if self.param_space.setup_params:
            yaml_dict['setup'] = {
                'factors': sorted(self.param_space.setup_params.keys()),
                'levels': {
                    name: ",".join(map(str, param.values))
                    for name, param in self.param_space.setup_params.items()
                },
                'treatments': setup_treatments
            }

        if self.param_space.run_params:
            yaml_dict['run'] = {
                'factors': sorted(self.param_space.run_params.keys()),
                'levels': {
                    name: ",".join(map(str, param.values))
                    for name, param in self.param_space.run_params.items()
                },
                'treatments': run_treatments
            }

        # Write to file
        with open(output_path, 'w') as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)

        logger.info(f"Exported {len(treatments)} treatments to {output_path}")

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'param_space': {
                'setup_params': len(self.param_space.setup_params),
                'run_params': len(self.param_space.run_params),
                'total_params': len(self.param_space.all_params),
                'encoded_dimension': self.param_space.encoded_dimension
            },
            'treatments': {
                'generated': len(self.generated_treatments),
                'feasible': len(self.feasible_treatments),
                'infeasible': len(self.infeasible_treatments),
                'feasibility_rate': (
                    len(self.feasible_treatments) / len(self.generated_treatments)
                    if self.generated_treatments else 0
                )
            },
            'validators': len(self.validators)
        }

    def __repr__(self) -> str:
        return (
            f"TreatmentGenerator("
            f"params={len(self.param_space.all_params)}, "
            f"treatments={len(self.generated_treatments)}, "
            f"feasible={len(self.feasible_treatments)})"
        )

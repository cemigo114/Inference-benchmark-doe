"""
Treatment validators for constraint checking and feasibility validation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import os
import sys
import logging

from .treatment import Treatment

logger = logging.getLogger(__name__)


class Validator(ABC):
    """Base class for treatment validators."""

    @abstractmethod
    def validate(self, treatment: Treatment) -> Tuple[bool, str]:
        """
        Validate a treatment.

        Returns:
            (is_valid, reason): True if valid, False otherwise with reason
        """
        pass


class ConstraintValidator(Validator):
    """
    Validates treatments against mathematical constraints.

    Examples:
        - replicas * tensor_parallelism <= total_gpus
        - max_model_len * block_size <= gpu_memory
    """

    def __init__(self, constraints: Optional[List[str]] = None):
        """
        Initialize with list of constraint expressions.

        Args:
            constraints: List of Python expressions that should evaluate to True
                Example: ["REPLICAS * TP <= 8", "MAX_MODEL_LEN < 32000"]
        """
        self.constraints = constraints or []

    def add_constraint(self, expression: str):
        """Add a constraint expression."""
        self.constraints.append(expression)

    def validate(self, treatment: Treatment) -> Tuple[bool, str]:
        """Validate treatment against all constraints."""
        # Create namespace with treatment parameters
        namespace = treatment.params.copy()

        # Add common constants
        namespace['NA'] = None
        namespace['None'] = None

        for i, constraint in enumerate(self.constraints):
            try:
                # Evaluate constraint
                result = eval(constraint, {"__builtins__": {}}, namespace)

                if not result:
                    return False, f"Constraint violated: {constraint}"

            except NameError as e:
                # Missing parameter - skip this constraint
                logger.debug(f"Skipping constraint {constraint}: {e}")
                continue

            except Exception as e:
                logger.warning(f"Error evaluating constraint '{constraint}': {e}")
                continue

        return True, ""


class CapacityPlannerValidator(Validator):
    """
    Validates treatments using the Capacity Planner.

    Checks:
    - Model fits in GPU memory with given TP
    - KV cache fits with given max_model_len
    - Sufficient total GPUs available
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        gpu_memory_gb: Optional[int] = None,
        total_gpus: Optional[int] = None,
        strict: bool = False
    ):
        """
        Initialize capacity planner validator.

        Args:
            model_name: Model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            gpu_memory_gb: GPU memory in GB (e.g., 80 for H100)
            total_gpus: Total GPUs available in cluster
            strict: If True, reject treatments that fail validation
                    If False, warn but allow (useful for testing)
        """
        self.model_name = model_name
        self.gpu_memory_gb = gpu_memory_gb
        self.total_gpus = total_gpus
        self.strict = strict

        # Try to import capacity planner
        self._capacity_planner_available = False
        try:
            # Add config_explorer to path if needed
            config_explorer_path = os.path.join(
                os.path.dirname(__file__),
                '../../config_explorer/src'
            )
            if os.path.exists(config_explorer_path):
                sys.path.insert(0, config_explorer_path)

            from config_explorer.capacity_planner import validate_config
            self._validate_config = validate_config
            self._capacity_planner_available = True

        except ImportError as e:
            logger.warning(
                f"Capacity Planner not available: {e}. "
                "Validation will be skipped."
            )

    def validate(self, treatment: Treatment) -> Tuple[bool, str]:
        """Validate treatment using Capacity Planner."""
        if not self._capacity_planner_available:
            if self.strict:
                return False, "Capacity Planner not available"
            else:
                logger.debug("Skipping Capacity Planner validation (not available)")
                return True, ""

        # Extract relevant parameters
        params = treatment.params

        # Determine deployment method
        deploy_method = params.get('LLMDBENCH_DEPLOY_METHODS', 'modelservice')

        # Build config for capacity planner
        try:
            config = self._build_capacity_planner_config(params, deploy_method)

            # Run validation
            is_valid = self._validate_config(config)

            if is_valid:
                return True, ""
            else:
                reason = "Capacity Planner validation failed: configuration not feasible"
                return False, reason

        except Exception as e:
            error_msg = f"Capacity Planner error: {str(e)}"
            logger.warning(error_msg)

            if self.strict:
                return False, error_msg
            else:
                # In non-strict mode, allow treatments that can't be validated
                return True, ""

    def _build_capacity_planner_config(
        self,
        params: Dict[str, Any],
        deploy_method: str
    ) -> Dict:
        """Build configuration dict for Capacity Planner."""
        # Use model from params or fallback to instance variable
        model = params.get('LLMDBENCH_DEPLOY_MODEL_LIST', self.model_name)

        config = {
            'model': model,
            'gpu_memory_gb': self.gpu_memory_gb,
        }

        if deploy_method == 'standalone':
            config.update({
                'tensor_parallelism': params.get('LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM', 1),
                'replicas': params.get('LLMDBENCH_VLLM_COMMON_REPLICAS', 1),
            })

        elif deploy_method == 'modelservice':
            config.update({
                'prefill_replicas': params.get('LLMDBENCH_VLLM_MODELSERVICE_PREFILL_REPLICAS', 0),
                'prefill_tp': params.get('LLMDBENCH_VLLM_MODELSERVICE_PREFILL_TENSOR_PARALLELISM', 1),
                'decode_replicas': params.get('LLMDBENCH_VLLM_MODELSERVICE_DECODE_REPLICAS', 1),
                'decode_tp': params.get('LLMDBENCH_VLLM_MODELSERVICE_DECODE_TENSOR_PARALLELISM', 1),
            })

        # Add optional parameters
        if 'LLMDBENCH_VLLM_COMMON_MAX_MODEL_LEN' in params:
            config['max_model_len'] = params['LLMDBENCH_VLLM_COMMON_MAX_MODEL_LEN']

        if 'LLMDBENCH_VLLM_COMMON_BLOCK_SIZE' in params:
            config['block_size'] = params['LLMDBENCH_VLLM_COMMON_BLOCK_SIZE']

        return config


class CompositeValidator(Validator):
    """
    Combines multiple validators.

    Runs all validators and aggregates results.
    """

    def __init__(self, validators: Optional[List[Validator]] = None):
        """Initialize with list of validators."""
        self.validators = validators or []

    def add_validator(self, validator: Validator):
        """Add a validator."""
        self.validators.append(validator)

    def validate(self, treatment: Treatment) -> Tuple[bool, str]:
        """Run all validators."""
        reasons = []

        for validator in self.validators:
            is_valid, reason = validator.validate(treatment)

            if not is_valid:
                reasons.append(reason)

        if reasons:
            return False, "; ".join(reasons)

        return True, ""


class TotalGPUValidator(Validator):
    """
    Validates that total GPU requirement doesn't exceed available GPUs.
    """

    def __init__(self, total_gpus: int):
        """
        Initialize validator.

        Args:
            total_gpus: Total GPUs available in cluster
        """
        self.total_gpus = total_gpus

    def validate(self, treatment: Treatment) -> Tuple[bool, str]:
        """Validate GPU requirement."""
        params = treatment.params
        deploy_method = params.get('LLMDBENCH_DEPLOY_METHODS', 'modelservice')

        required_gpus = 0

        if deploy_method == 'standalone':
            tp = params.get('LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM', 1)
            replicas = params.get('LLMDBENCH_VLLM_COMMON_REPLICAS', 1)
            required_gpus = tp * replicas

        elif deploy_method == 'modelservice':
            prefill_tp = params.get('LLMDBENCH_VLLM_MODELSERVICE_PREFILL_TENSOR_PARALLELISM', 1)
            prefill_replicas = params.get('LLMDBENCH_VLLM_MODELSERVICE_PREFILL_REPLICAS', 0)
            decode_tp = params.get('LLMDBENCH_VLLM_MODELSERVICE_DECODE_TENSOR_PARALLELISM', 1)
            decode_replicas = params.get('LLMDBENCH_VLLM_MODELSERVICE_DECODE_REPLICAS', 1)

            required_gpus = (prefill_tp * prefill_replicas) + (decode_tp * decode_replicas)

        if required_gpus > self.total_gpus:
            return False, (
                f"Requires {required_gpus} GPUs but only {self.total_gpus} available"
            )

        return True, ""


class NAValueValidator(Validator):
    """
    Validates that NA values are used correctly based on deployment method.

    For standalone: prefill/decode params should be NA
    For modelservice: common params should be NA
    """

    def validate(self, treatment: Treatment) -> Tuple[bool, str]:
        """Validate NA values."""
        params = treatment.params
        deploy_method = params.get('LLMDBENCH_DEPLOY_METHODS')

        if deploy_method == 'standalone':
            # Prefill/decode params should be NA
            prefill_decode_params = [
                'LLMDBENCH_VLLM_MODELSERVICE_PREFILL_REPLICAS',
                'LLMDBENCH_VLLM_MODELSERVICE_PREFILL_TENSOR_PARALLELISM',
                'LLMDBENCH_VLLM_MODELSERVICE_DECODE_REPLICAS',
                'LLMDBENCH_VLLM_MODELSERVICE_DECODE_TENSOR_PARALLELISM',
            ]

            for param in prefill_decode_params:
                if param in params and params[param] not in [None, 'NA']:
                    return False, (
                        f"Standalone deployment should have {param}=NA, "
                        f"got {params[param]}"
                    )

        elif deploy_method == 'modelservice':
            # Common params should be NA (except those shared)
            common_params = [
                'LLMDBENCH_VLLM_COMMON_REPLICAS',
                'LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM',
            ]

            for param in common_params:
                if param in params and params[param] not in [None, 'NA']:
                    return False, (
                        f"Modelservice deployment should have {param}=NA, "
                        f"got {params[param]}"
                    )

        return True, ""

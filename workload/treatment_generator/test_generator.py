"""
Unit tests for treatment generator module.
"""

import sys
from pathlib import Path

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from workload.treatment_generator.parameter_space import (
    ParameterSpace,
    Parameter,
    ParameterType,
)
from workload.treatment_generator.treatment import Treatment
from workload.treatment_generator.strategies import (
    FullFactorialStrategy,
    RandomSamplingStrategy,
    LatinHypercubeSamplingStrategy,
    SobolSequenceStrategy,
)
from workload.treatment_generator.validators import (
    ConstraintValidator,
    TotalGPUValidator,
    NAValueValidator,
)
from workload.treatment_generator.generator import TreatmentGenerator


class TestParameter:
    """Test Parameter class."""

    def test_categorical_parameter(self):
        param = Parameter(
            name="test_param",
            type=ParameterType.CATEGORICAL,
            values=["a", "b", "c"]
        )

        assert param.is_valid("a")
        assert param.is_valid("b")
        assert not param.is_valid("d")

    def test_ordinal_parameter(self):
        param = Parameter(
            name="test_param",
            type=ParameterType.ORDINAL,
            values=[1, 2, 4, 8],
            min_value=1,
            max_value=8
        )

        assert param.is_valid(1)
        assert param.is_valid(4)
        assert not param.is_valid(3)  # Not in values

    def test_continuous_parameter(self):
        param = Parameter(
            name="test_param",
            type=ParameterType.CONTINUOUS,
            min_value=0.0,
            max_value=1.0
        )

        assert param.is_valid(0.5)
        assert not param.is_valid(1.5)

    def test_normalize_denormalize(self):
        param = Parameter(
            name="test_param",
            type=ParameterType.ORDINAL,
            min_value=100,
            max_value=1000
        )

        # Normalize
        assert param.normalize(100) == 0.0
        assert param.normalize(1000) == 1.0
        assert abs(param.normalize(550) - 0.5) < 0.01

        # Denormalize
        assert param.denormalize(0.0) == 100
        assert param.denormalize(1.0) == 1000

    def test_infer_from_values(self):
        # Infer categorical
        param = Parameter.infer_from_values("test", ["a", "b", "c"])
        assert param.type == ParameterType.CATEGORICAL

        # Infer ordinal
        param = Parameter.infer_from_values("test", [1, 2, 4, 8])
        assert param.type == ParameterType.ORDINAL

        # Infer continuous
        param = Parameter.infer_from_values("test", [0.1, 0.5, 0.9])
        assert param.type == ParameterType.CONTINUOUS


class TestParameterSpace:
    """Test ParameterSpace class."""

    def test_add_parameter(self):
        space = ParameterSpace()

        param = Parameter(
            name="test_param",
            type=ParameterType.CATEGORICAL,
            values=["a", "b"]
        )

        space.add_parameter(param, phase="setup")
        assert "test_param" in space.setup_params

        space.add_parameter(param, phase="run")
        assert "test_param" in space.run_params

    def test_encode_decode(self):
        space = ParameterSpace()

        # Add categorical parameter
        space.add_parameter(
            Parameter(
                name="cat_param",
                type=ParameterType.CATEGORICAL,
                values=["a", "b", "c"]
            ),
            phase="setup"
        )

        # Add ordinal parameter
        space.add_parameter(
            Parameter(
                name="ord_param",
                type=ParameterType.ORDINAL,
                min_value=0,
                max_value=10
            ),
            phase="run"
        )

        # Test encoding
        treatment = {"cat_param": "b", "ord_param": 5}
        encoded = space.encode_treatment(treatment)

        # Should have 3 (one-hot) + 1 (normalized) = 4 dimensions
        assert len(encoded) == 4

        # Categorical one-hot: [0, 1, 0]
        assert encoded[0] == 0
        assert encoded[1] == 1
        assert encoded[2] == 0

        # Ordinal normalized: 5/10 = 0.5
        assert abs(encoded[3] - 0.5) < 0.01

        # Test decoding
        decoded = space.decode_vector(encoded)
        assert decoded["cat_param"] == "b"
        assert abs(decoded["ord_param"] - 5) < 1  # Within rounding

    def test_from_yaml_dict_parses_numeric_levels(self):
        yaml_dict = {
            "setup": {
                "factors": ["replicas"],
                "levels": {"replicas": "1,2,4"},
            },
            "run": {
                "factors": ["learning_rate"],
                "levels": {"learning_rate": "0.1,0.2,0.5"},
            },
        }

        space = ParameterSpace.from_yaml_dict(yaml_dict)

        replicas = space.setup_params["replicas"]
        assert replicas.type == ParameterType.ORDINAL
        assert replicas.values == [1, 2, 4]

        lr = space.run_params["learning_rate"]
        assert lr.type == ParameterType.CONTINUOUS
        assert lr.values == [0.1, 0.2, 0.5]


class TestTreatment:
    """Test Treatment class."""

    def test_create_treatment(self):
        treatment = Treatment(
            params={"param1": 1, "param2": "a"}
        )

        assert treatment.treatment_id is not None
        assert len(treatment.treatment_id) == 12

    def test_csv_conversion(self):
        param_names = ["param1", "param2", "param3"]
        csv_str = "1,2,3"

        treatment = Treatment.from_csv_string(csv_str, param_names)

        assert treatment.params["param1"] == 1
        assert treatment.params["param2"] == 2
        assert treatment.params["param3"] == 3

    def test_treatment_equality(self):
        t1 = Treatment(params={"a": 1, "b": 2})
        t2 = Treatment(params={"a": 1, "b": 2})
        t3 = Treatment(params={"a": 1, "b": 3})

        assert t1 == t2
        assert t1 != t3


class TestStrategies:
    """Test generation strategies."""

    @pytest.fixture
    def simple_param_space(self):
        space = ParameterSpace()

        space.add_parameter(
            Parameter(
                name="param1",
                type=ParameterType.ORDINAL,
                values=[1, 2, 4, 8]
            ),
            phase="setup"
        )

        space.add_parameter(
            Parameter(
                name="param2",
                type=ParameterType.ORDINAL,
                values=[10, 20, 30]
            ),
            phase="run"
        )

        return space

    def test_full_factorial(self, simple_param_space):
        strategy = FullFactorialStrategy()
        treatments = strategy.generate(simple_param_space)

        # 4 * 3 = 12 combinations
        assert len(treatments) == 12

        # Check uniqueness
        unique_treatments = set(t.treatment_id for t in treatments)
        assert len(unique_treatments) == 12

    def test_random_sampling(self, simple_param_space):
        strategy = RandomSamplingStrategy()
        treatments = strategy.generate(
            simple_param_space,
            n_samples=20,
            seed=42
        )

        assert len(treatments) == 20

        # All parameters should be valid
        for treatment in treatments:
            assert treatment.params["param1"] in [1, 2, 4, 8]
            assert treatment.params["param2"] in [10, 20, 30]

    def test_lhs_sampling(self, simple_param_space):
        strategy = LatinHypercubeSamplingStrategy()
        treatments = strategy.generate(
            simple_param_space,
            n_samples=10,
            seed=42
        )

        assert len(treatments) == 10

    def test_sobol_sampling(self, simple_param_space):
        strategy = SobolSequenceStrategy()
        treatments = strategy.generate(
            simple_param_space,
            n_samples=10,
            seed=42
        )

        assert len(treatments) == 10


class TestValidators:
    """Test validators."""

    def test_constraint_validator(self):
        validator = ConstraintValidator([
            "param1 * param2 <= 100",
            "param1 > 0"
        ])

        # Valid treatment
        t1 = Treatment(params={"param1": 5, "param2": 10})
        is_valid, reason = validator.validate(t1)
        assert is_valid

        # Invalid treatment (5 * 30 = 150 > 100)
        t2 = Treatment(params={"param1": 5, "param2": 30})
        is_valid, reason = validator.validate(t2)
        assert not is_valid

    def test_total_gpu_validator(self):
        validator = TotalGPUValidator(total_gpus=8)

        # Valid standalone treatment
        t1 = Treatment(params={
            "LLMDBENCH_DEPLOY_METHODS": "standalone",
            "LLMDBENCH_VLLM_COMMON_REPLICAS": 2,
            "LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM": 4
        })
        is_valid, reason = validator.validate(t1)
        assert is_valid  # 2 * 4 = 8 GPUs

        # Invalid standalone treatment
        t2 = Treatment(params={
            "LLMDBENCH_DEPLOY_METHODS": "standalone",
            "LLMDBENCH_VLLM_COMMON_REPLICAS": 4,
            "LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM": 4
        })
        is_valid, reason = validator.validate(t2)
        assert not is_valid  # 4 * 4 = 16 GPUs > 8

    def test_na_value_validator(self):
        validator = NAValueValidator()

        # Valid standalone (prefill/decode should be NA)
        t1 = Treatment(params={
            "LLMDBENCH_DEPLOY_METHODS": "standalone",
            "LLMDBENCH_VLLM_MODELSERVICE_PREFILL_REPLICAS": None
        })
        is_valid, reason = validator.validate(t1)
        assert is_valid

        # Invalid standalone (prefill should be NA)
        t2 = Treatment(params={
            "LLMDBENCH_DEPLOY_METHODS": "standalone",
            "LLMDBENCH_VLLM_MODELSERVICE_PREFILL_REPLICAS": 4
        })
        is_valid, reason = validator.validate(t2)
        assert not is_valid


class TestTreatmentGenerator:
    """Test TreatmentGenerator class."""

    def test_basic_generation(self):
        param_space = ParameterSpace()
        param_space.add_parameter(
            Parameter(
                name="param1",
                type=ParameterType.ORDINAL,
                values=[1, 2, 4]
            ),
            phase="setup"
        )

        generator = TreatmentGenerator(param_space=param_space)

        treatments = generator.generate(
            strategy='random',
            n_samples=10,
            validate=False
        )

        assert len(treatments) == 10
        assert len(generator.generated_treatments) == 10

    def test_with_validation(self):
        param_space = ParameterSpace()
        param_space.add_parameter(
            Parameter(
                name="replicas",
                type=ParameterType.ORDINAL,
                values=[1, 2, 4, 8]
            ),
            phase="setup"
        )
        param_space.add_parameter(
            Parameter(
                name="tp",
                type=ParameterType.ORDINAL,
                values=[1, 2, 4, 8]
            ),
            phase="setup"
        )

        generator = TreatmentGenerator(param_space=param_space)
        generator.add_validator(
            ConstraintValidator(["replicas * tp <= 8"])
        )

        treatments = generator.generate(
            strategy='full_factorial',
            validate=True
        )

        # Should filter out invalid combinations
        assert len(treatments) < 16  # Less than full factorial

        # All should satisfy constraint
        for treatment in treatments:
            assert treatment.params["replicas"] * treatment.params["tp"] <= 8

    def test_summary(self):
        param_space = ParameterSpace()
        param_space.add_parameter(
            Parameter(name="p1", type=ParameterType.ORDINAL, values=[1, 2]),
            phase="setup"
        )

        generator = TreatmentGenerator(param_space=param_space)
        generator.generate(strategy='random', n_samples=10, validate=False)

        summary = generator.summary()

        assert summary['param_space']['total_params'] == 1
        assert summary['treatments']['generated'] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

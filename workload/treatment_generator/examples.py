"""
Usage examples for TreatmentGenerator.

Demonstrates various ways to use the treatment generator module.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from treatment_generator import (
    TreatmentGenerator,
    ParameterSpace,
    Parameter,
    ParameterType,
)


def example_1_from_existing_yaml():
    """
    Example 1: Load from existing YAML format (backwards compatible).
    """
    print("=" * 70)
    print("Example 1: Loading from existing YAML format")
    print("=" * 70)

    # Load from old-style YAML
    yaml_path = "../../experiments/inference-scheduling.yaml"

    if os.path.exists(yaml_path):
        generator = TreatmentGenerator.from_yaml_file(yaml_path)

        print(f"\nParameter Space: {generator.param_space}")
        print(f"Setup parameters: {list(generator.param_space.setup_params.keys())}")
        print(f"Run parameters: {list(generator.param_space.run_params.keys())}")

        # Generate treatments using LHS instead of manual specification
        treatments = generator.generate(
            strategy='lhs',
            n_samples=10,
            validate=False
        )

        print(f"\nGenerated {len(treatments)} treatments using LHS")
        for i, treatment in enumerate(treatments[:3], 1):
            print(f"\nTreatment {i}:")
            for key, value in sorted(treatment.params.items()):
                print(f"  {key}: {value}")

    else:
        print(f"File not found: {yaml_path}")


def example_2_programmatic_definition():
    """
    Example 2: Define parameter space programmatically.
    """
    print("\n" + "=" * 70)
    print("Example 2: Programmatic parameter space definition")
    print("=" * 70)

    # Create parameter space
    param_space = ParameterSpace()

    # Add setup parameters
    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_VLLM_COMMON_REPLICAS",
            type=ParameterType.ORDINAL,
            values=[1, 2, 4, 8],
            description="Number of replicas"
        ),
        phase="setup"
    )

    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM",
            type=ParameterType.ORDINAL,
            values=[1, 2, 4, 8],
            description="Tensor parallelism degree"
        ),
        phase="setup"
    )

    # Add run parameters
    param_space.add_parameter(
        Parameter(
            name="max_concurrency",
            type=ParameterType.ORDINAL,
            values=[1, 8, 32, 64],
            description="Maximum concurrent requests"
        ),
        phase="run"
    )

    # Create generator
    generator = TreatmentGenerator(param_space=param_space)

    # Generate with different strategies
    print("\n--- Strategy: Full Factorial ---")
    try:
        factorial_treatments = generator.generate(
            strategy='full_factorial',
            validate=False,
            max_treatments=100
        )
        print(f"Generated {len(factorial_treatments)} treatments")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Strategy: Latin Hypercube Sampling ---")
    lhs_treatments = generator.generate(
        strategy='lhs',
        n_samples=16,
        validate=False
    )
    print(f"Generated {len(lhs_treatments)} treatments")

    print("\n--- Strategy: Sobol Sequence ---")
    sobol_treatments = generator.generate(
        strategy='sobol',
        n_samples=16,
        validate=False
    )
    print(f"Generated {len(sobol_treatments)} treatments")

    # Show first treatment from each strategy
    print("\nFirst LHS treatment:", lhs_treatments[0].params)
    print("First Sobol treatment:", sobol_treatments[0].params)


def example_3_with_validation():
    """
    Example 3: Generate treatments with validation.
    """
    print("\n" + "=" * 70)
    print("Example 3: Treatment generation with validation")
    print("=" * 70)

    from treatment_generator.validators import (
        ConstraintValidator,
        TotalGPUValidator,
        NAValueValidator
    )

    # Create parameter space
    param_space = ParameterSpace()

    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_DEPLOY_METHODS",
            type=ParameterType.CATEGORICAL,
            values=["standalone", "modelservice"]
        ),
        phase="setup"
    )

    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_VLLM_COMMON_REPLICAS",
            type=ParameterType.ORDINAL,
            values=[1, 2, 4, 8]
        ),
        phase="setup"
    )

    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM",
            type=ParameterType.ORDINAL,
            values=[1, 2, 4, 8]
        ),
        phase="setup"
    )

    # Create generator
    generator = TreatmentGenerator(param_space=param_space)

    # Add validators
    generator.add_validator(TotalGPUValidator(total_gpus=8))

    generator.add_validator(
        ConstraintValidator([
            "LLMDBENCH_VLLM_COMMON_REPLICAS * LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM <= 8"
        ])
    )

    # Generate with validation
    print("\nGenerating 50 random treatments...")
    treatments = generator.generate(
        strategy='random',
        n_samples=50,
        validate=True,
        seed=42
    )

    print(f"\nGenerated {len(treatments)} feasible treatments")
    print(f"Infeasible treatments: {len(generator.infeasible_treatments)}")

    # Show summary
    summary = generator.summary()
    print(f"\nSummary:")
    print(f"  Feasibility rate: {summary['treatments']['feasibility_rate']:.2%}")

    # Show some infeasible treatments
    if generator.infeasible_treatments:
        print(f"\nExample infeasible treatment:")
        infeasible = generator.infeasible_treatments[0]
        print(f"  Params: {infeasible.params}")
        print(f"  Reason: {infeasible.feasibility_reason}")


def example_4_export_to_legacy_format():
    """
    Example 4: Export generated treatments to legacy YAML format.
    """
    print("\n" + "=" * 70)
    print("Example 4: Export to legacy YAML format")
    print("=" * 70)

    # Create simple parameter space
    param_space = ParameterSpace()

    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_VLLM_MODELSERVICE_GAIE_PLUGINS_CONFIGFILE",
            type=ParameterType.CATEGORICAL,
            values=["inf-sche-none.yaml", "inf-sche-prefix.yaml"]
        ),
        phase="setup"
    )

    param_space.add_parameter(
        Parameter(
            name="question_len",
            type=ParameterType.ORDINAL,
            values=[100, 500, 1000]
        ),
        phase="run"
    )

    param_space.add_parameter(
        Parameter(
            name="output_len",
            type=ParameterType.ORDINAL,
            values=[100, 500, 1000]
        ),
        phase="run"
    )

    # Generate treatments
    generator = TreatmentGenerator(param_space=param_space)
    treatments = generator.generate(
        strategy='lhs',
        n_samples=8,
        validate=False,
        seed=42
    )

    # Export to legacy format
    output_path = "/tmp/generated_experiment.yaml"
    generator.to_legacy_yaml(treatments, output_path)

    print(f"\nExported to: {output_path}")
    print("\nGenerated YAML content:")
    with open(output_path, 'r') as f:
        print(f.read())


def example_5_new_yaml_format():
    """
    Example 5: Use new YAML format with improved features.
    """
    print("\n" + "=" * 70)
    print("Example 5: New YAML format with metadata and constraints")
    print("=" * 70)

    yaml_path = "../../experiments/inference-scheduling-v2.yaml"

    if os.path.exists(yaml_path):
        generator = TreatmentGenerator.from_yaml_file(yaml_path)

        print(f"\nParameter Space: {generator.param_space}")

        # Generate treatments
        treatments = generator.generate(
            strategy='lhs',
            n_samples=12,
            validate=True
        )

        print(f"\nGenerated {len(treatments)} treatments")

        # Show summary
        summary = generator.summary()
        print(f"\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    else:
        print(f"File not found: {yaml_path}")


def example_6_comparison_strategies():
    """
    Example 6: Compare different sampling strategies.
    """
    print("\n" + "=" * 70)
    print("Example 6: Compare sampling strategies")
    print("=" * 70)

    # Simple 2D parameter space for visualization
    param_space = ParameterSpace()

    param_space.add_parameter(
        Parameter(
            name="question_len",
            type=ParameterType.ORDINAL,
            min_value=100,
            max_value=1000,
            values=[100, 300, 500, 700, 1000]
        ),
        phase="run"
    )

    param_space.add_parameter(
        Parameter(
            name="output_len",
            type=ParameterType.ORDINAL,
            min_value=100,
            max_value=1000,
            values=[100, 300, 500, 700, 1000]
        ),
        phase="run"
    )

    strategies = ['random', 'lhs', 'sobol', 'grid']
    n_samples = 10

    for strategy in strategies:
        generator = TreatmentGenerator(param_space=param_space)

        if strategy == 'grid':
            treatments = generator.generate(
                strategy=strategy,
                n_points_per_dim=3,
                validate=False
            )
        else:
            treatments = generator.generate(
                strategy=strategy,
                n_samples=n_samples,
                validate=False,
                seed=42
            )

        print(f"\n--- {strategy.upper()} ({len(treatments)} treatments) ---")

        # Show parameter distribution
        question_lens = [t.params['question_len'] for t in treatments]
        output_lens = [t.params['output_len'] for t in treatments]

        print(f"question_len: min={min(question_lens)}, max={max(question_lens)}, "
              f"unique={len(set(question_lens))}")
        print(f"output_len: min={min(output_lens)}, max={max(output_lens)}, "
              f"unique={len(set(output_lens))}")


if __name__ == "__main__":
    # Run all examples
    examples = [
        example_1_from_existing_yaml,
        example_2_programmatic_definition,
        example_3_with_validation,
        example_4_export_to_legacy_format,
        example_5_new_yaml_format,
        example_6_comparison_strategies,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)

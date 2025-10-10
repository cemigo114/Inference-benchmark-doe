"""
Examples demonstrating two-stage experimental design.

Shows how to use categorical screening + Bayesian optimization
for more rigorous experimental design.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from treatment_generator import (
    ParameterSpace,
    Parameter,
    ParameterType,
    Treatment
)
from treatment_generator.two_stage_design import (
    TwoStageTreatmentGenerator,
    CategoricalScreening,
    BayesianOptimizationStage
)


def example_1_basic_two_stage():
    """
    Example 1: Basic two-stage design with categorical screening + LHS.
    """
    print("=" * 70)
    print("Example 1: Basic Two-Stage Design")
    print("=" * 70)

    # Create parameter space with mixed types
    param_space = ParameterSpace()

    # Categorical parameters (deployment configuration)
    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_DEPLOY_METHODS",
            type=ParameterType.CATEGORICAL,
            values=["standalone", "modelservice"],
            description="Deployment method"
        ),
        phase="setup"
    )

    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_VLLM_MODELSERVICE_GAIE_PLUGINS_CONFIGFILE",
            type=ParameterType.CATEGORICAL,
            values=["inf-sche-none.yaml", "inf-sche-prefix.yaml", "inf-sche-kv.yaml"],
            description="GAIE scheduling policy"
        ),
        phase="setup"
    )

    # Numerical parameters (workload configuration)
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
            name="max_concurrency",
            type=ParameterType.ORDINAL,
            values=[1, 8, 32, 64, 128],
            description="Max concurrent requests"
        ),
        phase="run"
    )

    # Initialize two-stage generator
    generator = TwoStageTreatmentGenerator(param_space)

    # Stage 1: Categorical screening with pairwise covering
    print("\n--- Stage 1: Categorical Screening ---")
    categorical_configs = generator.stage1_screening(method='pairwise')

    print(f"Generated {len(categorical_configs)} categorical configurations")
    for i, config in enumerate(categorical_configs[:3], 1):
        print(f"\nConfig {i}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # Stage 2: LHS for each categorical config
    print("\n--- Stage 2: LHS for Numerical Parameters ---")

    all_treatments = []
    for i, config in enumerate(categorical_configs[:2], 1):  # Just first 2 for demo
        print(f"\nGenerating treatments for Config {i}...")

        treatments = generator.stage2_initial(
            config,
            n_samples=5,
            criterion='maximin'
        )

        print(f"  Generated {len(treatments)} treatments")
        print(f"  Sample treatment:")
        print(f"    {treatments[0].params}")

        all_treatments.extend(treatments)

    print(f"\nTotal treatments: {len(all_treatments)}")


def example_2_compare_screening_methods():
    """
    Example 2: Compare different categorical screening methods.
    """
    print("\n" + "=" * 70)
    print("Example 2: Compare Categorical Screening Methods")
    print("=" * 70)

    # Create parameter space with 3 categorical factors
    param_space = ParameterSpace()

    param_space.add_parameter(
        Parameter(
            name="deployment",
            type=ParameterType.CATEGORICAL,
            values=["standalone", "modelservice"]
        ),
        phase="setup"
    )

    param_space.add_parameter(
        Parameter(
            name="scheduler",
            type=ParameterType.CATEGORICAL,
            values=["none", "prefix", "kv", "queue"]
        ),
        phase="setup"
    )

    param_space.add_parameter(
        Parameter(
            name="model_size",
            type=ParameterType.CATEGORICAL,
            values=["small", "medium", "large"]
        ),
        phase="setup"
    )

    # Add one numerical parameter
    param_space.add_parameter(
        Parameter(
            name="replicas",
            type=ParameterType.ORDINAL,
            values=[1, 2, 4, 8]
        ),
        phase="setup"
    )

    generator = TwoStageTreatmentGenerator(param_space)

    methods = ['pairwise', 'full']

    for method in methods:
        print(f"\n--- Method: {method.upper()} ---")

        configs = generator.stage1_screening(method=method)

        print(f"Categorical configurations: {len(configs)}")

        if method == 'pairwise':
            # Calculate coverage
            pairs_covered = set()
            for config in configs:
                for k1, v1 in config.items():
                    for k2, v2 in config.items():
                        if k1 < k2:
                            pairs_covered.add(f"{k1}={v1}|{k2}={v2}")

            print(f"Unique pairs covered: {len(pairs_covered)}")

            # Total possible pairs
            total_pairs = 0
            params = [p for p in param_space.all_params.values()
                     if p.type == ParameterType.CATEGORICAL]
            for i, p1 in enumerate(params):
                for p2 in params[i+1:]:
                    total_pairs += len(p1.values) * len(p2.values)

            print(f"Total possible pairs: {total_pairs}")
            print(f"Coverage: {len(pairs_covered)/total_pairs:.1%}")

        # Show sample configs
        print("\nSample configurations:")
        for i, config in enumerate(configs[:3], 1):
            print(f"  {i}. {config}")


def example_3_bayesian_optimization_workflow():
    """
    Example 3: Complete Bayesian optimization workflow with simulated objectives.
    """
    print("\n" + "=" * 70)
    print("Example 3: Bayesian Optimization Workflow")
    print("=" * 70)

    # Create simple parameter space
    param_space = ParameterSpace()

    # One categorical
    param_space.add_parameter(
        Parameter(
            name="scheduler",
            type=ParameterType.CATEGORICAL,
            values=["none", "prefix"]
        ),
        phase="setup"
    )

    # Two numerical
    param_space.add_parameter(
        Parameter(
            name="replicas",
            type=ParameterType.ORDINAL,
            min_value=1,
            max_value=8,
            values=[1, 2, 4, 8]
        ),
        phase="setup"
    )

    param_space.add_parameter(
        Parameter(
            name="concurrency",
            type=ParameterType.ORDINAL,
            min_value=1,
            max_value=128,
            values=[1, 8, 32, 64, 128]
        ),
        phase="run"
    )

    generator = TwoStageTreatmentGenerator(param_space)

    # Stage 1: Get categorical configs
    categorical_configs = generator.stage1_screening(method='full')
    print(f"Categorical configs: {len(categorical_configs)}")

    # For each categorical config, run Bayesian optimization
    for config in categorical_configs:
        print(f"\n--- Optimizing for: {config} ---")

        # Stage 2a: Initial LHS
        initial_treatments = generator.stage2_initial(
            config,
            n_samples=5
        )

        print(f"Initial LHS: {len(initial_treatments)} samples")

        # Simulate evaluating treatments
        def simulate_objective(treatment: Treatment) -> float:
            """
            Simulate throughput objective.

            Simple model: throughput ≈ f(replicas, concurrency)
            Peak at replicas=4, concurrency=64
            """
            replicas = treatment.params.get('replicas', 1)
            concurrency = treatment.params.get('concurrency', 1)
            scheduler = treatment.params.get('scheduler', 'none')

            # Base throughput
            throughput = 100 * replicas * np.log1p(concurrency)

            # Peak around replicas=4
            throughput *= np.exp(-0.1 * (replicas - 4) ** 2)

            # Peak around concurrency=64
            throughput *= np.exp(-0.001 * (concurrency - 64) ** 2)

            # Scheduler boost
            if scheduler == 'prefix':
                throughput *= 1.2

            # Add noise
            throughput += np.random.randn() * 10

            return throughput

        # Evaluate initial samples
        objectives = [simulate_objective(t) for t in initial_treatments]

        print(f"Initial objectives: min={min(objectives):.1f}, "
              f"max={max(objectives):.1f}, mean={np.mean(objectives):.1f}")

        # Update BO model
        generator.stage2_update(config, initial_treatments, objectives)

        # Bayesian optimization loop
        n_iterations = 5

        for iteration in range(n_iterations):
            print(f"\nBO Iteration {iteration + 1}/{n_iterations}")

            # Suggest next treatment
            next_treatment = generator.stage2_suggest(
                config,
                acquisition='ei'
            )

            # Evaluate
            objective = simulate_objective(next_treatment)

            print(f"  Suggested: replicas={next_treatment.params.get('replicas')}, "
                  f"concurrency={next_treatment.params.get('concurrency')}")
            print(f"  Objective: {objective:.1f}")

            # Update model
            generator.stage2_update(config, [next_treatment], [objective])

            objectives.append(objective)

        # Show best found
        best_idx = np.argmax(objectives)
        all_treatments = initial_treatments + [next_treatment]  # Simplified
        best_treatment = initial_treatments[min(best_idx, len(initial_treatments)-1)]

        print(f"\nBest treatment found:")
        print(f"  Params: {best_treatment.params}")
        print(f"  Objective: {max(objectives):.1f}")


def example_4_fractional_factorial_screening():
    """
    Example 4: Fractional factorial design for categorical screening.
    """
    print("\n" + "=" * 70)
    print("Example 4: Fractional Factorial Screening")
    print("=" * 70)

    # Create parameter space with many 2-level categorical factors
    param_space = ParameterSpace()

    factors = [
        ("enable_prefix_cache", ["true", "false"]),
        ("enable_chunked_prefill", ["true", "false"]),
        ("enable_kv_sharing", ["true", "false"]),
        ("enable_speculative_decoding", ["true", "false"]),
        ("use_v2_block_manager", ["true", "false"]),
    ]

    for name, values in factors:
        param_space.add_parameter(
            Parameter(
                name=name,
                type=ParameterType.CATEGORICAL,
                values=values
            ),
            phase="setup"
        )

    # Add numerical parameter
    param_space.add_parameter(
        Parameter(
            name="batch_size",
            type=ParameterType.ORDINAL,
            values=[1, 2, 4, 8, 16]
        ),
        phase="setup"
    )

    generator = TwoStageTreatmentGenerator(param_space)

    # Compare full vs fractional factorial
    print("\n--- Full Factorial ---")
    full_configs = generator.stage1_screening(method='full')
    print(f"Configurations: {len(full_configs)}")
    print(f"Total runs needed: {len(full_configs)} × batch_size levels")

    print("\n--- Fractional Factorial (Resolution IV) ---")
    frac_configs = generator.stage1_screening(
        method='fractional',
        resolution=4
    )
    print(f"Configurations: {len(frac_configs)}")
    print(f"Total runs needed: {len(frac_configs)} × batch_size levels")
    print(f"Reduction: {len(full_configs)/len(frac_configs):.1f}x fewer")

    print("\nSample fractional factorial configurations:")
    for i, config in enumerate(frac_configs[:5], 1):
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        print(f"  {i}. {config_str}")


def example_5_complete_two_stage_workflow():
    """
    Example 5: Complete workflow generating full design.
    """
    print("\n" + "=" * 70)
    print("Example 5: Complete Two-Stage Workflow")
    print("=" * 70)

    # Realistic LLM-D parameter space
    param_space = ParameterSpace()

    # Categorical: deployment + scheduler
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
            name="LLMDBENCH_VLLM_MODELSERVICE_GAIE_PLUGINS_CONFIGFILE",
            type=ParameterType.CATEGORICAL,
            values=["inf-sche-none.yaml", "inf-sche-prefix.yaml"]
        ),
        phase="setup"
    )

    # Numerical: infrastructure
    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_VLLM_COMMON_REPLICAS",
            type=ParameterType.ORDINAL,
            min_value=1,
            max_value=8,
            values=[1, 2, 4, 8]
        ),
        phase="setup"
    )

    param_space.add_parameter(
        Parameter(
            name="LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM",
            type=ParameterType.ORDINAL,
            min_value=1,
            max_value=8,
            values=[1, 2, 4, 8]
        ),
        phase="setup"
    )

    # Numerical: workload
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

    # Generate complete design
    generator = TwoStageTreatmentGenerator(param_space)

    treatments = generator.generate_full_design(
        stage1_method='pairwise',
        stage2_samples_per_config=8,
        stage2_criterion='maximin'
    )

    print(f"\nGenerated {len(treatments)} total treatments")

    # Analyze design
    categorical_configs = {}
    for treatment in treatments:
        config_key = (
            treatment.params.get('LLMDBENCH_DEPLOY_METHODS'),
            treatment.params.get('LLMDBENCH_VLLM_MODELSERVICE_GAIE_PLUGINS_CONFIGFILE')
        )
        categorical_configs[config_key] = categorical_configs.get(config_key, 0) + 1

    print(f"\nCategorical configurations tested: {len(categorical_configs)}")
    for config, count in categorical_configs.items():
        print(f"  {config}: {count} treatments")

    # Show sample treatments
    print("\nSample treatments:")
    for i, treatment in enumerate(treatments[:3], 1):
        print(f"\n  Treatment {i}:")
        for key, value in sorted(treatment.params.items()):
            print(f"    {key}: {value}")


if __name__ == "__main__":
    examples = [
        example_1_basic_two_stage,
        example_2_compare_screening_methods,
        example_3_bayesian_optimization_workflow,
        example_4_fractional_factorial_screening,
        example_5_complete_two_stage_workflow,
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

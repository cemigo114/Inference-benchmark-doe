#!/usr/bin/env python3
"""
Command-line interface for treatment generator.

Usage:
    python -m treatment_generator.cli generate experiment.yaml --strategy lhs --n-samples 20
    python -m treatment_generator.cli validate experiment.yaml
    python -m treatment_generator.cli compare experiment.yaml --strategies lhs,sobol,random
"""

import argparse
import sys
import json
from pathlib import Path

from generator import TreatmentGenerator


def cmd_generate(args):
    """Generate treatments from experiment YAML."""
    print(f"Loading experiment from: {args.input}")

    generator = TreatmentGenerator.from_yaml_file(args.input)

    print(f"\nParameter space:")
    print(f"  Setup parameters: {len(generator.param_space.setup_params)}")
    print(f"  Run parameters: {len(generator.param_space.run_params)}")

    print(f"\nGenerating treatments...")
    print(f"  Strategy: {args.strategy}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Validate: {args.validate}")

    treatments = generator.generate(
        strategy=args.strategy,
        n_samples=args.n_samples,
        validate=args.validate,
        seed=args.seed
    )

    print(f"\nGenerated {len(treatments)} treatments")

    if args.validate:
        summary = generator.summary()
        print(f"\nValidation results:")
        print(f"  Feasible: {summary['treatments']['feasible']}")
        print(f"  Infeasible: {summary['treatments']['infeasible']}")
        print(f"  Feasibility rate: {summary['treatments']['feasibility_rate']:.2%}")

    # Export
    if args.format == 'legacy':
        generator.to_legacy_yaml(treatments, args.output)
        print(f"\nExported to legacy YAML: {args.output}")

    elif args.format == 'json':
        output = {
            'metadata': {
                'strategy': args.strategy,
                'n_samples': args.n_samples,
            },
            'treatments': [t.to_dict() for t in treatments]
        }

        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nExported to JSON: {args.output}")

    # Show sample treatments
    if args.show_samples > 0:
        print(f"\nSample treatments:")
        for i, treatment in enumerate(treatments[:args.show_samples], 1):
            print(f"\n  Treatment {i} ({treatment.treatment_id}):")
            for key, value in sorted(treatment.params.items()):
                print(f"    {key}: {value}")


def cmd_validate(args):
    """Validate treatments in experiment YAML."""
    print(f"Loading experiment from: {args.input}")

    generator = TreatmentGenerator.from_yaml_file(args.input)

    # Load existing treatments
    import yaml
    with open(args.input, 'r') as f:
        config = yaml.safe_load(f)

    # Extract explicit treatments if present
    setup_treatments = config.get('setup', {}).get('treatments', [])
    run_treatments = config.get('run', {}).get('treatments', [])

    if not setup_treatments and not run_treatments:
        print("No explicit treatments found. Generating...")
        treatments = generator.generate(
            strategy='lhs',
            n_samples=20,
            validate=True
        )
    else:
        print(f"Found {len(setup_treatments)} setup treatments")
        treatments = generator.generate_from_explicit_list(
            setup_treatments,
            validate=True
        )

    # Show validation results
    summary = generator.summary()

    print(f"\nValidation Results:")
    print(f"  Total treatments: {summary['treatments']['generated']}")
    print(f"  Feasible: {summary['treatments']['feasible']}")
    print(f"  Infeasible: {summary['treatments']['infeasible']}")
    print(f"  Feasibility rate: {summary['treatments']['feasibility_rate']:.2%}")

    # Show infeasible treatments
    if generator.infeasible_treatments:
        print(f"\nInfeasible treatments:")
        for treatment in generator.infeasible_treatments[:10]:
            print(f"\n  {treatment.treatment_id}:")
            print(f"    Params: {treatment.params}")
            print(f"    Reason: {treatment.feasibility_reason}")

    # Export report
    if args.report:
        report = {
            'summary': summary,
            'feasible_treatments': [t.to_dict() for t in generator.feasible_treatments],
            'infeasible_treatments': [t.to_dict() for t in generator.infeasible_treatments]
        }

        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {args.report}")


def cmd_compare(args):
    """Compare different sampling strategies."""
    print(f"Loading experiment from: {args.input}")

    strategies = args.strategies.split(',')

    print(f"\nComparing strategies: {strategies}")
    print(f"Samples per strategy: {args.n_samples}")

    results = {}

    for strategy in strategies:
        print(f"\n--- {strategy.upper()} ---")

        generator = TreatmentGenerator.from_yaml_file(args.input)

        treatments = generator.generate(
            strategy=strategy.strip(),
            n_samples=args.n_samples,
            validate=args.validate,
            seed=args.seed
        )

        summary = generator.summary()

        results[strategy] = {
            'treatments': len(treatments),
            'feasible': summary['treatments']['feasible'],
            'infeasible': summary['treatments']['infeasible'],
            'feasibility_rate': summary['treatments']['feasibility_rate']
        }

        print(f"  Generated: {len(treatments)}")
        print(f"  Feasible: {summary['treatments']['feasible']}")
        print(f"  Feasibility rate: {summary['treatments']['feasibility_rate']:.2%}")

    # Summary table
    print(f"\n{'Strategy':<15} {'Generated':<12} {'Feasible':<12} {'Rate':<10}")
    print("-" * 55)
    for strategy, data in results.items():
        print(f"{strategy:<15} {data['treatments']:<12} {data['feasible']:<12} "
              f"{data['feasibility_rate']:.2%}")

    # Export comparison
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nComparison saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='Treatment Generator CLI for LLM-D Benchmark'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Generate command
    parser_gen = subparsers.add_parser('generate', help='Generate treatments')
    parser_gen.add_argument('input', help='Input experiment YAML file')
    parser_gen.add_argument('--output', '-o', default='generated_treatments.yaml',
                           help='Output file (default: generated_treatments.yaml)')
    parser_gen.add_argument('--strategy', '-s', default='lhs',
                           choices=['lhs', 'sobol', 'random', 'grid', 'full_factorial'],
                           help='Sampling strategy (default: lhs)')
    parser_gen.add_argument('--n-samples', '-n', type=int, default=20,
                           help='Number of samples (default: 20)')
    parser_gen.add_argument('--validate', action='store_true',
                           help='Validate treatments')
    parser_gen.add_argument('--seed', type=int, default=None,
                           help='Random seed for reproducibility')
    parser_gen.add_argument('--format', choices=['legacy', 'json'], default='legacy',
                           help='Output format (default: legacy)')
    parser_gen.add_argument('--show-samples', type=int, default=3,
                           help='Number of sample treatments to display (default: 3)')
    parser_gen.set_defaults(func=cmd_generate)

    # Validate command
    parser_val = subparsers.add_parser('validate', help='Validate treatments')
    parser_val.add_argument('input', help='Input experiment YAML file')
    parser_val.add_argument('--report', '-r', help='Output validation report (JSON)')
    parser_val.set_defaults(func=cmd_validate)

    # Compare command
    parser_cmp = subparsers.add_parser('compare', help='Compare sampling strategies')
    parser_cmp.add_argument('input', help='Input experiment YAML file')
    parser_cmp.add_argument('--strategies', default='lhs,sobol,random',
                           help='Comma-separated strategies (default: lhs,sobol,random)')
    parser_cmp.add_argument('--n-samples', '-n', type=int, default=20,
                           help='Number of samples per strategy (default: 20)')
    parser_cmp.add_argument('--validate', action='store_true',
                           help='Validate treatments')
    parser_cmp.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    parser_cmp.add_argument('--output', '-o', help='Output comparison report (JSON)')
    parser_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    try:
        args.func(args)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

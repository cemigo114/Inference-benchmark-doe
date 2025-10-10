# Treatment Generator - 5-Minute Quickstart

## Installation

```bash
cd llm-d-benchmark
pip install -r workload/treatment_generator/requirements.txt
```

## Example 1: Generate from Existing YAML (Backwards Compatible)

```python
from workload.treatment_generator import TreatmentGenerator

# Load existing experiment YAML
generator = TreatmentGenerator.from_yaml_file(
    "experiments/inference-scheduling.yaml"
)

# Generate 20 treatments using Latin Hypercube Sampling
treatments = generator.generate(
    strategy='lhs',
    n_samples=20,
    validate=True
)

print(f"Generated {len(treatments)} treatments")

# Show first treatment
print(treatments[0].params)
```

**Output:**
```
Generated 20 treatments
{
    'LLMDBENCH_VLLM_MODELSERVICE_GAIE_PLUGINS_CONFIGFILE': 'inf-sche-prefix.yaml',
    'question_len': 425,
    'output_len': 672
}
```

## Example 2: Export to Legacy Format

```python
# Generate treatments
generator = TreatmentGenerator.from_yaml_file(
    "experiments/inference-scheduling.yaml"
)
treatments = generator.generate(strategy='lhs', n_samples=15)

# Export to legacy YAML for e2e.sh
generator.to_legacy_yaml(
    treatments,
    "experiments/generated_treatments.yaml"
)

print("Exported to: experiments/generated_treatments.yaml")
```

**Then run with e2e.sh:**
```bash
./e2e.sh --experiments generated_treatments.yaml
```

## Example 3: Use New YAML Format

Create `my_experiment.yaml`:
```yaml
metadata:
  name: "Quick Test Experiment"

cluster:
  total_gpus: 8
  gpu_memory_gb: 80

setup:
  parameters:
    - name: LLMDBENCH_VLLM_COMMON_REPLICAS
      type: ordinal
      values: [1, 2, 4]

    - name: LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM
      type: ordinal
      values: [1, 2, 4]

run:
  parameters:
    - name: max_concurrency
      type: ordinal
      values: [1, 8, 32]

generation:
  strategy: lhs
  n_samples: 10

constraints:
  expressions:
    - "LLMDBENCH_VLLM_COMMON_REPLICAS * LLMDBENCH_VLLM_COMMON_TENSOR_PARALLELISM <= 8"
```

**Generate:**
```python
generator = TreatmentGenerator.from_yaml_file("my_experiment.yaml")
treatments = generator.generate(strategy='lhs', n_samples=10, validate=True)

# Check validation results
summary = generator.summary()
print(f"Feasible: {summary['treatments']['feasible']}")
print(f"Infeasible: {summary['treatments']['infeasible']}")
```

## Example 4: Command-Line Usage

```bash
# Generate treatments
python -m workload.treatment_generator.cli generate \
    experiments/inference-scheduling.yaml \
    --strategy lhs \
    --n-samples 20 \
    --validate \
    --output generated.yaml

# Validate existing experiment
python -m workload.treatment_generator.cli validate \
    experiments/pd-disaggregation.yaml \
    --report validation_report.json

# Compare strategies
python -m workload.treatment_generator.cli compare \
    experiments/inference-scheduling.yaml \
    --strategies lhs,sobol,random \
    --n-samples 20
```

## Example 5: Programmatic Definition

```python
from workload.treatment_generator import (
    TreatmentGenerator,
    ParameterSpace,
    Parameter,
    ParameterType
)

# Create parameter space
param_space = ParameterSpace()

# Add parameters
param_space.add_parameter(
    Parameter(
        name="replicas",
        type=ParameterType.ORDINAL,
        values=[1, 2, 4, 8],
        description="Number of replicas"
    ),
    phase="setup"
)

param_space.add_parameter(
    Parameter(
        name="concurrency",
        type=ParameterType.ORDINAL,
        values=[1, 8, 32, 64],
        description="Max concurrent requests"
    ),
    phase="run"
)

# Create generator
generator = TreatmentGenerator(param_space=param_space)

# Add constraint validator
from workload.treatment_generator.validators import ConstraintValidator
generator.add_validator(
    ConstraintValidator(["replicas <= 8", "concurrency >= 1"])
)

# Generate treatments
treatments = generator.generate(
    strategy='lhs',
    n_samples=15,
    validate=True,
    seed=42  # For reproducibility
)

print(f"Generated {len(treatments)} feasible treatments")
```

## Common Tasks

### Task 1: Reduce Full Factorial to Manageable Size

**Before:**
```yaml
# 4 × 3 × 3 = 36 combinations
treatments:
  - "none,100,100"
  - "none,100,300"
  # ... 34 more lines
```

**After:**
```python
# Use LHS to get 12 representative samples
generator = TreatmentGenerator.from_yaml_file("experiment.yaml")
treatments = generator.generate(strategy='lhs', n_samples=12)
```

### Task 2: Validate Before Running

```python
# Generate with validation
generator = TreatmentGenerator.from_yaml_file("experiment.yaml")
treatments = generator.generate(validate=True)

# Check what failed
for t in generator.infeasible_treatments[:5]:
    print(f"Invalid: {t.params}")
    print(f"Reason: {t.feasibility_reason}\n")
```

### Task 3: Compare Sampling Strategies

```python
strategies = ['lhs', 'sobol', 'random']

for strategy in strategies:
    gen = TreatmentGenerator.from_yaml_file("experiment.yaml")
    treatments = gen.generate(strategy=strategy, n_samples=20, seed=42)

    print(f"\n{strategy.upper()}:")
    print(f"  Unique treatments: {len(set(t.treatment_id for t in treatments))}")

    # Check parameter coverage
    values = [t.params['question_len'] for t in treatments]
    print(f"  question_len range: {min(values)}-{max(values)}")
```

### Task 4: Export for Integration

```python
# Generate treatments
generator = TreatmentGenerator.from_yaml_file("experiment.yaml")
treatments = generator.generate(strategy='lhs', n_samples=20)

# Export to legacy YAML (for e2e.sh)
generator.to_legacy_yaml(treatments, "output/experiment.yaml")

# Also export to JSON (for other tools)
import json
with open("output/treatments.json", 'w') as f:
    json.dump([t.to_dict() for t in treatments], f, indent=2)
```

## Troubleshooting

### Issue: "No module named treatment_generator"

**Solution:** Make sure you're in the llm-d-benchmark directory:
```bash
cd llm-d-benchmark
python -c "from workload.treatment_generator import TreatmentGenerator"
```

### Issue: "Capacity Planner not available"

**Solution:** This is normal. Capacity Planner integration is optional:
```python
# Will warn but continue
generator = TreatmentGenerator.from_yaml_file("experiment.yaml")
treatments = generator.generate(validate=True)
```

### Issue: All treatments marked infeasible

**Solution:** Check constraints:
```python
# See what's failing
for t in generator.infeasible_treatments[:3]:
    print(f"Reason: {t.feasibility_reason}")

# Relax constraints or adjust parameter ranges
```

### Issue: Want to see more examples

**Solution:** Run the examples file:
```bash
cd llm-d-benchmark
python workload/treatment_generator/examples.py
```

## Next Steps

1. **Read full documentation**: `workload/treatment_generator/README.md`
2. **Run examples**: `python workload/treatment_generator/examples.py`
3. **Run tests**: `pytest workload/treatment_generator/test_generator.py -v`
4. **Review design**: `workload/treatment_generator/DESIGN.md`

## Cheat Sheet

```python
# Import
from workload.treatment_generator import TreatmentGenerator

# Load YAML
gen = TreatmentGenerator.from_yaml_file("experiment.yaml")

# Generate (choose one strategy)
gen.generate(strategy='lhs', n_samples=20)      # Recommended
gen.generate(strategy='sobol', n_samples=20)    # Deterministic
gen.generate(strategy='random', n_samples=20)   # Baseline
gen.generate(strategy='full_factorial')         # All combinations

# Validate
gen.generate(validate=True)  # Filter infeasible

# Export
gen.to_legacy_yaml(treatments, "output.yaml")

# Summary
print(gen.summary())
```

## Get Help

- **Documentation**: `workload/treatment_generator/README.md`
- **Examples**: `workload/treatment_generator/examples.py`
- **Tests**: `workload/treatment_generator/test_generator.py`
- **Issues**: https://github.com/llm-d/llm-d-benchmark/issues

**Happy experimenting! 🚀**

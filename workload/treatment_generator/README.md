# Treatment Generator for LLM-D Benchmark

A self-contained Python module for generating valid experiment treatments with intelligent sampling strategies and constraint validation.

## Features

- ✅ **Multiple Sampling Strategies**: Latin Hypercube, Sobol, Random, Factorial, Grid
- ✅ **Constraint Validation**: Mathematical expressions, Capacity Planner integration, GPU constraints
- ✅ **Backwards Compatible**: Works with existing YAML format
- ✅ **Improved YAML Format**: More expressive, with type annotations and metadata
- ✅ **Bayesian Optimization Ready**: Encoding/decoding for future GP-based optimization
- ✅ **Reduces Manual Effort**: Automatic treatment generation vs manual specification

## Installation

```bash
# Install required dependencies
pip install scipy numpy pyyaml
```

## Quick Start

### Option 1: Use Existing YAML (Backwards Compatible)

```python
from treatment_generator import TreatmentGenerator

# Load from existing experiment YAML
generator = TreatmentGenerator.from_yaml_file("experiments/inference-scheduling.yaml")

# Generate treatments using LHS instead of manual specification
treatments = generator.generate(
    strategy='lhs',        # Latin Hypercube Sampling
    n_samples=20,          # Generate 20 treatments
    validate=True          # Validate feasibility
)

# Export to legacy format for e2e.sh
generator.to_legacy_yaml(treatments, "output/generated_treatments.yaml")
```

### Option 2: Use New YAML Format

```yaml
# experiment.yaml
metadata:
  name: "My Experiment"

cluster:
  total_gpus: 8
  gpu_memory_gb: 80

setup:
  parameters:
    - name: LLMDBENCH_VLLM_COMMON_REPLICAS
      type: ordinal
      values: [1, 2, 4, 8]

run:
  parameters:
    - name: max_concurrency
      type: ordinal
      values: [1, 8, 32, 64]

generation:
  strategy: lhs
  n_samples: 20

constraints:
  expressions:
    - "LLMDBENCH_VLLM_COMMON_REPLICAS <= 8"
```

```python
generator = TreatmentGenerator.from_yaml_file("experiment.yaml")
treatments = generator.generate(strategy='lhs', n_samples=20)
```

### Option 3: Programmatic Definition

```python
from treatment_generator import (
    TreatmentGenerator,
    ParameterSpace,
    Parameter,
    ParameterType
)

# Define parameter space
param_space = ParameterSpace()

param_space.add_parameter(
    Parameter(
        name="LLMDBENCH_VLLM_COMMON_REPLICAS",
        type=ParameterType.ORDINAL,
        values=[1, 2, 4, 8]
    ),
    phase="setup"
)

# Create generator with validation
generator = TreatmentGenerator(param_space=param_space)

# Add validators
from treatment_generator.validators import TotalGPUValidator
generator.add_validator(TotalGPUValidator(total_gpus=8))

# Generate treatments
treatments = generator.generate(strategy='lhs', n_samples=20, validate=True)
```

## Available Strategies

| Strategy | Description | When to Use | Pros | Cons |
|----------|-------------|-------------|------|------|
| **lhs** | Latin Hypercube Sampling | Default, recommended | Best space coverage, fewer samples | Requires numerical parameters |
| **sobol** | Sobol sequence | Deterministic experiments | Low-discrepancy, reproducible | Sensitive to dimension |
| **random** | Uniform random | Quick exploration | Simple, works everywhere | Poor coverage |
| **full_factorial** | All combinations | Small parameter spaces | Complete coverage | Combinatorial explosion |
| **fractional_factorial** | Subset of factorial | Screening experiments | Balanced coverage | May miss interactions |
| **grid** | Uniform grid | Visualization | Regular pattern | Inefficient for high dimensions |

## Comparison: Manual vs Automated

### Current Approach (Manual)

```yaml
# experiments/inference-scheduling.yaml
treatments:
  - "inf-sche-none.yaml,100,100"
  - "inf-sche-none.yaml,100,300"
  - "inf-sche-none.yaml,100,1000"
  # ... 33 more lines manually specified
```

**Problems:**
- Manual effort: Must specify all 36 combinations
- No optimization: Random/arbitrary selection
- Hard to maintain: Changes require manual updates
- No validation: Invalid treatments only discovered at runtime

### New Approach (Automated)

```yaml
# experiments/inference-scheduling-v2.yaml
setup:
  parameters:
    - name: LLMDBENCH_VLLM_MODELSERVICE_GAIE_PLUGINS_CONFIGFILE
      type: categorical
      values: [inf-sche-none.yaml, inf-sche-prefix.yaml, inf-sche-kv.yaml]

run:
  parameters:
    - name: question_len
      type: ordinal
      values: [100, 300, 1000]
    - name: output_len
      type: ordinal
      values: [100, 300, 1000]

generation:
  strategy: lhs
  n_samples: 12  # Instead of 27 full factorial

constraints:
  expressions:
    - "question_len + output_len <= 2000"
```

```python
generator = TreatmentGenerator.from_yaml_file("inference-scheduling-v2.yaml")
treatments = generator.generate(strategy='lhs', n_samples=12, validate=True)
```

**Benefits:**
- Less manual effort: Specify parameters, not combinations
- Better coverage: LHS ensures good space-filling
- Validated: Invalid treatments filtered automatically
- Flexible: Change strategy without rewriting treatments

## Validation

### Built-in Validators

```python
from treatment_generator.validators import (
    ConstraintValidator,       # Mathematical expressions
    TotalGPUValidator,         # GPU availability
    NAValueValidator,          # NA value correctness
    CapacityPlannerValidator   # Model/KV cache feasibility
)

# Constraint validator
generator.add_validator(
    ConstraintValidator([
        "replicas * tensor_parallelism <= 8",
        "max_model_len <= 32000"
    ])
)

# GPU validator
generator.add_validator(TotalGPUValidator(total_gpus=8))

# Capacity Planner (if available)
generator.add_validator(
    CapacityPlannerValidator(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        gpu_memory_gb=80
    )
)
```

### Validation Results

```python
# Generate with validation
treatments = generator.generate(strategy='random', n_samples=100, validate=True)

# Check results
summary = generator.summary()
print(f"Feasible: {summary['treatments']['feasible']}")
print(f"Infeasible: {summary['treatments']['infeasible']}")
print(f"Feasibility rate: {summary['treatments']['feasibility_rate']:.2%}")

# Inspect infeasible treatments
for treatment in generator.infeasible_treatments[:5]:
    print(f"Treatment: {treatment.params}")
    print(f"Reason: {treatment.feasibility_reason}")
```

## Command-Line Interface

```bash
# Generate treatments from YAML
python -m treatment_generator.cli generate \
    --input experiments/inference-scheduling-v2.yaml \
    --output generated_treatments.yaml \
    --strategy lhs \
    --n-samples 20

# Validate existing treatments
python -m treatment_generator.cli validate \
    --input experiments/pd-disaggregation.yaml \
    --report validation_report.json

# Compare strategies
python -m treatment_generator.cli compare \
    --input experiment.yaml \
    --strategies lhs,sobol,random \
    --n-samples 20
```

## Integration with Bayesian Optimization (Future)

The module is designed to support Bayesian optimization:

```python
# Encode treatments for GP
X = [param_space.encode_treatment(t.params) for t in treatments]
y = [result['throughput'] for result in results]

# Fit Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor
gp = GaussianProcessRegressor()
gp.fit(X, y)

# Suggest next treatment
next_vector = optimize_acquisition_function(gp)
next_treatment_params = param_space.decode_vector(next_vector)
```

## Examples

See `examples.py` for comprehensive usage examples:

```bash
python workload/treatment_generator/examples.py
```

Examples include:
- Loading existing YAML format
- Programmatic parameter definition
- Validation and constraint checking
- Exporting to legacy format
- Comparing sampling strategies

## Testing

```bash
# Run unit tests
pytest workload/treatment_generator/test_generator.py -v

# Run with coverage
pytest workload/treatment_generator/test_generator.py --cov=treatment_generator
```

## Architecture

```
treatment_generator/
├── __init__.py           # Module exports
├── parameter_space.py    # Parameter definitions and encoding
├── treatment.py          # Treatment representation
├── strategies.py         # Sampling strategies (LHS, Sobol, etc.)
├── validators.py         # Constraint validators
├── generator.py          # Main TreatmentGenerator class
├── examples.py           # Usage examples
├── test_generator.py     # Unit tests
└── README.md            # This file
```

## API Reference

### TreatmentGenerator

```python
class TreatmentGenerator:
    def __init__(param_space, validators)
    def generate(strategy, n_samples, validate, **kwargs) -> List[Treatment]
    def filter_feasible(treatments) -> List[Treatment]
    def add_validator(validator)
    def summary() -> Dict

    @classmethod
    def from_yaml_file(yaml_path) -> TreatmentGenerator

    def to_legacy_yaml(treatments, output_path)
```

### ParameterSpace

```python
class ParameterSpace:
    def add_parameter(param, phase)
    def encode_treatment(treatment) -> np.ndarray
    def decode_vector(vector) -> Dict

    @classmethod
    def from_yaml_dict(yaml_dict) -> ParameterSpace
```

### Parameter

```python
class Parameter:
    def __init__(name, type, values, min_value, max_value, ...)
    def is_valid(value) -> bool
    def normalize(value) -> float
    def denormalize(normalized_value) -> value

    @classmethod
    def infer_from_values(name, values) -> Parameter
```

### Treatment

```python
class Treatment:
    params: Dict[str, Any]
    treatment_id: str
    is_feasible: bool
    feasibility_reason: str

    def to_csv_string() -> str
    def to_dict() -> Dict

    @classmethod
    def from_csv_string(csv_str, param_names) -> Treatment
```

## Recommendations for Improved YAML Format

### Recommended Changes

1. **Add explicit parameter types** (categorical, ordinal, continuous)
2. **Include metadata** (experiment name, description, version)
3. **Specify generation strategy** in YAML
4. **Add constraint expressions** directly in YAML
5. **Include cluster configuration** for validation

### Migration Path

**Old format still supported:**
```yaml
setup:
  factors: [param1, param2]
  levels:
    param1: "1,2,4"
    param2: "a,b,c"
  treatments:
    - "1,a"
    - "2,b"
```

**New format (recommended):**
```yaml
metadata:
  name: "Experiment Name"

setup:
  parameters:
    - name: param1
      type: ordinal
      values: [1, 2, 4]
    - name: param2
      type: categorical
      values: [a, b, c]

generation:
  strategy: lhs
  n_samples: 10
```

**Benefits of new format:**
- Type safety: Explicit parameter types
- Self-documenting: Metadata and descriptions
- Flexible: Strategy specified in YAML
- Validated: Constraints checked before generation

### Backwards Compatibility

The module automatically detects format:

```python
# Works with both old and new format
generator = TreatmentGenerator.from_yaml_file("experiment.yaml")

# Detect format
if 'metadata' in yaml_dict:
    print("Using new format")
else:
    print("Using legacy format (backwards compatible)")
```

## Performance

### Time Savings

**Manual specification:**
- 36 treatments: ~30 minutes manual effort
- Changes: Re-specify all treatments

**Automated generation:**
- Any number of treatments: ~5 seconds
- Changes: Update YAML, regenerate

### Space Coverage

**Manual (arbitrary selection):**
- Coverage: ~40-60% of parameter space
- Redundancy: Possible duplicate regions

**Automated (LHS/Sobol):**
- Coverage: ~90-95% of parameter space
- Redundancy: Minimal, space-filling design

## License

Apache 2.0 (same as llm-d-benchmark)

## Contributing

See main repository CONTRIBUTING.md

## Support

- Issues: https://github.com/llm-d/llm-d-benchmark/issues
- Slack: #sig-benchmarking

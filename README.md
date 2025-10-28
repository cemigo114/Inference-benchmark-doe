This is a research branch that implements a two-stage DOE framework to optimize LLM infrastructure configurations with minimal experimentation. There are 3 pieces of code to 1) simulate data 2) run DOE and fit models 3) create performance results and compare with baseline (random and grid sampling).

Note this is for demo/education purposes using simulated data; for real LLM-D benchmarks, it is recommended to
1. Generate pairwise+LHS treatment configurations, run benchmarks
2. Use real performance data to train GP, run BO-selected benchmarks


There are 3 python components:
1. modelsim.py: simulating benchmark data given the lack of real and balanced performance data. Use simple formulas for basic patterns of throughput, latency, and memory under common settings, for example:

2. two_stage_simulation.py 2-Stage DOE Methodology 
  Stage 1: Categorical Screening: Identify best categorical configuration with minimal evaluations

  2.1.1 Pairwise Coverage Algorithm 
    - Generates all 2-way combinations of categorical parameters
    - Uses greedy algorithm to find minimal covering array
    - For 3 parameters [deployment×scheduler×gaie_plugin_config]:
        - 2 × 3 × 3 = 18 full factorial combinations
      - Pairwise coverage reduces to ~6 evaluations
  2.1.2 Fixed Numerical Parameters 
  replicas = 4       # Middle of [1, 2, 4, 8]
  concurrency = 64   # Middle of [1, 8, 32, 64, 128, 256]
  batch_size = 8     # Middle of [1, 2, 4, 8, 16, 32]
  2.1.3. Evaluation
    - Each categorical combo evaluated once
    - Select top K (default: 1) by efficiency metric
  Result: Best categorical configuration(s) identified with ~6-9 evaluations

  Stage 2: Bayesian Optimization: Fine-tune numerical parameters for fixed categorical config
  
  2.2.1. Initial Exploration 
    - Latin Hypercube Sampling (LHS) for 5 initial points
    - Space-filling design over numerical parameter space
  2.2.2 Gaussian Process Surrogate 
  kernel = ConstantKernel × Matern(length_scale=1.0, nu=2.5)
  GP = GaussianProcessRegressor(kernel, normalize_y=True)
    - Matern kernel: good for non-smooth functions
    - Normalized Y values for stable training
  2.2.3. Expected Improvement Acquisition 
  EI(x) = (μ(x) - f_best) × Φ(Z) + σ(x) × φ(Z)
  where Z = (μ(x) - f_best) / σ(x)
    - μ(x): predicted mean at x
    - σ(x): predicted uncertainty at x
    - f_best: current best observed value
    - Balances exploitation (high μ) vs exploration (high σ)
  2.2.4. Sequential Optimization 
    - For each iteration (default: 20):
        i. Fit GP on observed data (numerical params only)
        ii. Generate 100 candidate configurations
       iii. Select next point maximizing EI
      iv. Evaluate and update GP training data
  Result: Optimal numerical parameters for each top categorical config



 3. formula_plots.py- Visualization and Results
 For repro: run all three .py files

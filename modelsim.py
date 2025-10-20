from typing import Dict
import math
from pprint import pprint
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


@dataclass
class PerformanceOptions:
    """Configuration options for performance simulation constants."""

    # Base performance constants (tunable based on hardware)
    BASE_THROUGHPUT_PER_REPLICA: float = 10.0  # requests/sec per replica at batch_size=1
    BASE_LATENCY: float = 5000.0  # ms base processing time
    BASE_MEMORY_PER_REPLICA: float = 8.0  # GB baseline memory per replica

    # Throughput scaling factors
    COORDINATION_OVERHEAD: float = 0.02  # logarithmic overhead for inter-replica coordination
    OPTIMAL_CONCURRENCY_PER_REPLICA: float = 4.0  # sweet spot for concurrent requests per replica
    UNDER_CAPACITY_BASE_EFFICIENCY: float = 0.8  # baseline efficiency when under capacity
    UNDER_CAPACITY_SCALE_FACTOR: float = 0.2  # additional efficiency gained as load increases
    OVER_CAPACITY_DEGRADATION: float = 0.3  # throughput degradation coefficient when over capacity
    BATCH_EFFICIENCY_THRESHOLD: float = 4.0  # batch size threshold for linear throughput scaling
    
    # Latency factors
    BATCH_WAIT_TIME_PER_ITEM: float = 1.0  # ms added per additional item in batch
    QUEUE_LATENCY_BASE: float = 10.0  # base queueing latency multiplier
    QUEUE_UTILIZATION_THRESHOLD: float = 0.9  # utilization level where queueing becomes exponential
    QUEUE_LATENCY_HEAVY_LOAD: float = 50.0  # base latency under heavy load
    QUEUE_EXPONENTIAL_FACTOR: float = 5.0  # exponential growth rate for queue under heavy load
    NETWORK_LATENCY_BASE: float = 2.0  # ms base network latency
    NETWORK_LATENCY_SCALE: float = 0.5  # logarithmic scaling factor for network latency
    COORDINATION_LATENCY_SCALE: float = 3.0  # logarithmic scaling factor for coordination latency
    MAX_UTILIZATION: float = 0.95  # maximum utilization to cap queueing calculations
    
    # Memory factors
    BATCH_MEMORY_INCREMENT: float = 0.15  # memory increase per additional batch item (15%)
    CONCURRENCY_MEMORY_PER_REQUEST: float = 0.1  # GB buffer memory per concurrent request
    CLUSTER_MEMORY_OVERHEAD_SCALE: float = 0.5  # logarithmic scaling for cluster coordination memory

    # GPU memory constraints
    MAX_GPU_MEMORY_PER_REPLICA: float = 32.0  # GB total GPU memory available per replica
    MODEL_MEMORY_OVERHEAD: float = 5.0  # GB memory for model weights and overhead
    MODEL_MEMORY_PER_REQUEST: float = 1.0  # GB memory per concurrent request (activations)


class UpdateMode(Enum):
    """How to apply a configuration value."""
    MULTIPLY = "multiply"  # Multiply the base value
    ADD = "add"           # Add to the base value
    SET = "set"           # Set absolute value (use sparingly)


class CategoricalSetting:
    """Base class for categorical configuration settings with delta-based updates."""
    
    def __init__(self, configs: Dict[int, Dict[str, tuple]]):
        """
        Initialize a categorical setting with configurations.
        
        Args:
            configs: Dictionary mapping indices (0, 1, 2) to configuration dictionaries.
                    Each config dict maps parameter names to (value, mode) tuples.
                    mode can be UpdateMode.MULTIPLY, UpdateMode.ADD, or UpdateMode.SET
        """
        self._configs = configs
    
    def __getitem__(self, index: int) -> Dict[str, tuple]:
        """Get configuration for a given index."""
        if index not in self._configs:
            raise IndexError(f"Invalid index {index}. Must be 0, 1, or 2.")
        return self._configs[index]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(configs with {len(self._configs)} variants)"


# Create instances with delta-based configurations

NETWORK_TOPOLOGY = CategoricalSetting({
    0: {  # Flat/Single-Datacenter - baseline (no changes from default)
        'COORDINATION_OVERHEAD': (1.0, UpdateMode.MULTIPLY),
        'NETWORK_LATENCY_BASE': (1.0, UpdateMode.MULTIPLY),
        'NETWORK_LATENCY_SCALE': (1.0, UpdateMode.MULTIPLY),
        'CLUSTER_MEMORY_OVERHEAD_SCALE': (1.0, UpdateMode.MULTIPLY)
    },
    1: {  # Multi-Zone - moderate increases
        'COORDINATION_OVERHEAD': (2.0, UpdateMode.MULTIPLY),
        'NETWORK_LATENCY_BASE': (2.5, UpdateMode.MULTIPLY),
        'NETWORK_LATENCY_SCALE': (2.4, UpdateMode.MULTIPLY),
        'CLUSTER_MEMORY_OVERHEAD_SCALE': (1.6, UpdateMode.MULTIPLY)
    },
    2: {  # Multi-Region - significant increases
        'COORDINATION_OVERHEAD': (4.0, UpdateMode.MULTIPLY),
        'NETWORK_LATENCY_BASE': (7.5, UpdateMode.MULTIPLY),
        'NETWORK_LATENCY_SCALE': (4.0, UpdateMode.MULTIPLY),
        'CLUSTER_MEMORY_OVERHEAD_SCALE': (2.4, UpdateMode.MULTIPLY)
    }
})


LOAD_BALANCER_STRATEGY = CategoricalSetting({
    0: {  # Round-Robin - slightly worse than baseline
        'OPTIMAL_CONCURRENCY_PER_REPLICA': (0.0, UpdateMode.ADD),
        'UNDER_CAPACITY_BASE_EFFICIENCY': (-0.05, UpdateMode.ADD),
        'QUEUE_UTILIZATION_THRESHOLD': (-0.05, UpdateMode.ADD),
        'QUEUE_LATENCY_BASE': (2.0, UpdateMode.ADD)
    },
    1: {  # Least-Connections - slightly better than baseline
        'OPTIMAL_CONCURRENCY_PER_REPLICA': (5.0, UpdateMode.ADD),
        'UNDER_CAPACITY_BASE_EFFICIENCY': (0.05, UpdateMode.ADD),
        'QUEUE_UTILIZATION_THRESHOLD': (0.0, UpdateMode.ADD),
        'QUEUE_LATENCY_BASE': (0.0, UpdateMode.ADD)
    },
    2: {  # Adaptive/AI-Based - best but with coordination cost
        'OPTIMAL_CONCURRENCY_PER_REPLICA': (10.0, UpdateMode.ADD),
        'UNDER_CAPACITY_BASE_EFFICIENCY': (0.1, UpdateMode.ADD),
        'QUEUE_UTILIZATION_THRESHOLD': (0.02, UpdateMode.ADD),
        'QUEUE_LATENCY_BASE': (-2.0, UpdateMode.ADD),
        'COORDINATION_LATENCY_SCALE': (2.0, UpdateMode.ADD)
    }
})


BATCHING_MODE = CategoricalSetting({
    0: {  # Static/Timeout-Based - slower batching
        'BATCH_WAIT_TIME_PER_ITEM': (1.6, UpdateMode.MULTIPLY),
        'BASE_THROUGHPUT_PER_REPLICA': (0.8, UpdateMode.MULTIPLY),
        'BATCH_MEMORY_INCREMENT': (0.8, UpdateMode.MULTIPLY)
    },
    1: {  # Dynamic/Opportunistic - baseline
        'BATCH_WAIT_TIME_PER_ITEM': (1.0, UpdateMode.MULTIPLY),
        'BASE_THROUGHPUT_PER_REPLICA': (1.0, UpdateMode.MULTIPLY),
        'BATCH_MEMORY_INCREMENT': (1.0, UpdateMode.MULTIPLY)
    },
    2: {  # Continuous-Batching - fast batching, reduced latency
        'BATCH_WAIT_TIME_PER_ITEM': (0.4, UpdateMode.MULTIPLY),
        'BASE_THROUGHPUT_PER_REPLICA': (1.4, UpdateMode.MULTIPLY),
        'BATCH_MEMORY_INCREMENT': (1.33, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (0.8, UpdateMode.MULTIPLY)
    }
})


AUTOSCALING_POLICY = CategoricalSetting({
    0: {  # No Autoscaling - severe degradation when overloaded
        'OVER_CAPACITY_DEGRADATION': (1.67, UpdateMode.MULTIPLY),
        'QUEUE_EXPONENTIAL_FACTOR': (1.6, UpdateMode.MULTIPLY),
        'MAX_UTILIZATION': (0.0, UpdateMode.ADD),
        'CONCURRENCY_MEMORY_PER_REQUEST': (1.5, UpdateMode.MULTIPLY)
    },
    1: {  # Reactive Scaling - baseline
        'OVER_CAPACITY_DEGRADATION': (1.0, UpdateMode.MULTIPLY),
        'QUEUE_EXPONENTIAL_FACTOR': (1.0, UpdateMode.MULTIPLY),
        'MAX_UTILIZATION': (-0.02, UpdateMode.ADD),
        'CONCURRENCY_MEMORY_PER_REQUEST': (1.0, UpdateMode.MULTIPLY)
    },
    2: {  # Predictive Scaling - graceful degradation
        'OVER_CAPACITY_DEGRADATION': (0.5, UpdateMode.MULTIPLY),
        'QUEUE_EXPONENTIAL_FACTOR': (0.6, UpdateMode.MULTIPLY),
        'MAX_UTILIZATION': (-0.05, UpdateMode.ADD),
        'CONCURRENCY_MEMORY_PER_REQUEST': (0.8, UpdateMode.MULTIPLY),
        'QUEUE_LATENCY_HEAVY_LOAD': (0.6, UpdateMode.MULTIPLY)
    }
})


HARDWARE_TIER = CategoricalSetting({
    0: {  # CPU-Only - poor performance
        'BASE_THROUGHPUT_PER_REPLICA': (0.5, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (2.4, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (0.5, UpdateMode.MULTIPLY),
        'OPTIMAL_CONCURRENCY_PER_REPLICA': (0.75, UpdateMode.MULTIPLY)
    },
    1: {  # Single GPU - baseline
        'BASE_THROUGHPUT_PER_REPLICA': (1.0, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (1.0, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (1.0, UpdateMode.MULTIPLY),
        'OPTIMAL_CONCURRENCY_PER_REPLICA': (1.0, UpdateMode.MULTIPLY)
    },
    2: {  # Multi-GPU/TPU - excellent performance
        'BASE_THROUGHPUT_PER_REPLICA': (2.5, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (0.4, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (3.0, UpdateMode.MULTIPLY),
        'OPTIMAL_CONCURRENCY_PER_REPLICA': (2.0, UpdateMode.MULTIPLY)
    }
})


MODEL_SERVING_FRAMEWORK = CategoricalSetting({
    0: {  # Basic/Flask-Based - inefficient
        'BASE_THROUGHPUT_PER_REPLICA': (0.6, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (1.6, UpdateMode.MULTIPLY),
        'UNDER_CAPACITY_BASE_EFFICIENCY': (-0.1, UpdateMode.ADD),
        'COORDINATION_OVERHEAD': (2.5, UpdateMode.MULTIPLY)
    },
    1: {  # Optimized/FastAPI+Triton - baseline
        'BASE_THROUGHPUT_PER_REPLICA': (1.0, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (1.0, UpdateMode.MULTIPLY),
        'UNDER_CAPACITY_BASE_EFFICIENCY': (0.0, UpdateMode.ADD),
        'COORDINATION_OVERHEAD': (1.0, UpdateMode.MULTIPLY)
    },
    2: {  # Specialized/vLLM+Ray - highly optimized
        'BASE_THROUGHPUT_PER_REPLICA': (1.5, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (0.7, UpdateMode.MULTIPLY),
        'UNDER_CAPACITY_BASE_EFFICIENCY': (0.1, UpdateMode.ADD),
        'COORDINATION_OVERHEAD': (1.5, UpdateMode.MULTIPLY),
        'BATCH_WAIT_TIME_PER_ITEM': (0.6, UpdateMode.MULTIPLY)
    }
})


CACHE_STRATEGY = CategoricalSetting({
    0: {  # No Caching - baseline
        'BASE_LATENCY': (1.0, UpdateMode.MULTIPLY),
        'BASE_THROUGHPUT_PER_REPLICA': (1.0, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (1.0, UpdateMode.MULTIPLY)
    },
    1: {  # Result Caching - moderate improvement
        'BASE_LATENCY': (0.9, UpdateMode.MULTIPLY),
        'BASE_THROUGHPUT_PER_REPLICA': (1.2, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (1.25, UpdateMode.MULTIPLY),
        'QUEUE_LATENCY_BASE': (0.8, UpdateMode.MULTIPLY)
    },
    2: {  # KV-Cache + Prefix Caching - significant improvement
        'BASE_LATENCY': (0.7, UpdateMode.MULTIPLY),
        'BASE_THROUGHPUT_PER_REPLICA': (1.6, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (1.75, UpdateMode.MULTIPLY),
        'QUEUE_LATENCY_BASE': (0.6, UpdateMode.MULTIPLY),
        'BATCH_MEMORY_INCREMENT': (0.67, UpdateMode.MULTIPLY)
    }
})


DEPLOYMENT = CategoricalSetting({
    0: {  # Standalone - baseline
        'BASE_THROUGHPUT_PER_REPLICA': (1.0, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (1.0, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (1.0, UpdateMode.MULTIPLY)
    },
    1: {  # ModelService - better performance and efficiency
        'BASE_THROUGHPUT_PER_REPLICA': (1.6, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (0.75, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (0.9, UpdateMode.MULTIPLY),
        'UNDER_CAPACITY_BASE_EFFICIENCY': (0.1, UpdateMode.ADD)
    },
    2: {  # Advanced/Distributed - best performance but with coordination costs
        'BASE_THROUGHPUT_PER_REPLICA': (2.0, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (0.6, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (0.85, UpdateMode.MULTIPLY),
        'UNDER_CAPACITY_BASE_EFFICIENCY': (0.15, UpdateMode.ADD),
        'COORDINATION_OVERHEAD': (1.5, UpdateMode.MULTIPLY)
    }
})


SCHEDULER = CategoricalSetting({
    0: {  # No Scheduler - baseline
        'BASE_THROUGHPUT_PER_REPLICA': (1.0, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (1.0, UpdateMode.MULTIPLY)
    },
    1: {  # Prefix Scheduler - good throughput boost with moderate latency
        'BASE_THROUGHPUT_PER_REPLICA': (1.3, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (1.1, UpdateMode.MULTIPLY),
        'BATCH_EFFICIENCY_THRESHOLD': (1.2, UpdateMode.MULTIPLY)
    },
    2: {  # KV Scheduler - balanced boost with higher latency penalty
        'BASE_THROUGHPUT_PER_REPLICA': (1.2, UpdateMode.MULTIPLY),
        'BASE_LATENCY': (1.2, UpdateMode.MULTIPLY),
        'BATCH_EFFICIENCY_THRESHOLD': (1.5, UpdateMode.MULTIPLY),
        'BASE_MEMORY_PER_REPLICA': (0.9, UpdateMode.MULTIPLY)
    }
})


# Constant return value for infeasible configurations
ZERO = {
    'throughput': 0.0,
    'latency': float('inf'),
    'max_memory_usage': 0.0,
    'utilization': float('inf'),
    'max_memory_usage_per_replica': 0.0,
    'max_batch_size_per_replica': 0
}


def simulate_performance(params: Dict, options: PerformanceOptions = None) -> Dict:
    """
    Simulate aggregate performance of a server cluster handling LLM inference requests.

    Args:
        params: Dictionary containing:
            - replicas: Number of server replicas in cluster
            - concurrency: Number of simultaneous client connections/requests
            - batch_size: Dynamic batch size for grouping requests on server
        options: PerformanceOptions instance with simulation constants (uses defaults if None)

    Returns:
        Dictionary containing:
            - throughput: Average achieved throughput (requests/second)
            - latency: Average latency in milliseconds
            - max_memory_usage: Peak memory usage in GB
    """
    if options is None:
        options = PerformanceOptions()

    replicas = params['replicas']
    concurrency = params['concurrency']
    batch_size = params['batch_size']

    # Calculate maximum batch size per replica based on GPU memory constraints
    max_batch_size_per_replica = max(0, math.floor(
        (options.MAX_GPU_MEMORY_PER_REPLICA - options.MODEL_MEMORY_OVERHEAD) / options.MODEL_MEMORY_PER_REQUEST
    ))

    # Check if batch_size exceeds replica capacity
    if batch_size > max_batch_size_per_replica:
        return ZERO
    
    # --- THROUGHPUT MODEL ---
    # Throughput scales with replicas but has diminishing returns due to coordination overhead
    replica_efficiency = 1.0 - (options.COORDINATION_OVERHEAD * math.log(1 + replicas))

    # Batching improves throughput linearly up to threshold, then plateaus
    # (thresholded model: linear scaling until BATCH_EFFICIENCY_THRESHOLD, constant after)
    batch_efficiency = min(batch_size, options.BATCH_EFFICIENCY_THRESHOLD)
    
    # High concurrency can saturate the cluster, causing contention
    concurrency_load_factor = concurrency / (replicas * options.OPTIMAL_CONCURRENCY_PER_REPLICA)
    
    if concurrency_load_factor <= 1.0:
        # Under capacity: near-linear scaling
        concurrency_efficiency = (options.UNDER_CAPACITY_BASE_EFFICIENCY + 
                                  options.UNDER_CAPACITY_SCALE_FACTOR * concurrency_load_factor)
    else:
        # Over capacity: throughput plateaus and degrades
        concurrency_efficiency = 1.0 / (1.0 + options.OVER_CAPACITY_DEGRADATION * (concurrency_load_factor - 1.0))
    
    throughput = (options.BASE_THROUGHPUT_PER_REPLICA * replicas * replica_efficiency * 
                  batch_efficiency * concurrency_efficiency)
    
    # --- LATENCY MODEL ---
    # Base latency for processing
    processing_latency = options.BASE_LATENCY
    
    # Batching adds latency due to waiting for batch to fill
    # Higher batch sizes mean longer waits on average
    batch_wait_time = (batch_size - 1) * options.BATCH_WAIT_TIME_PER_ITEM
    
    # Queueing delay increases with load (using M/M/c queue approximation)
    #utilization = min(options.MAX_UTILIZATION, concurrency_load_factor)
    utilization = concurrency_load_factor
    if utilization < options.QUEUE_UTILIZATION_THRESHOLD:
        queue_latency = options.QUEUE_LATENCY_BASE * utilization / (1.0 - utilization)
    else:
        # Heavy load: exponential growth in queue time
        excess_utilization = utilization - options.QUEUE_UTILIZATION_THRESHOLD
        queue_latency = (options.QUEUE_LATENCY_BASE + options.QUEUE_LATENCY_HEAVY_LOAD *
                            math.exp(min(options.QUEUE_EXPONENTIAL_FACTOR * excess_utilization, 100)))
    
    # Network latency between replicas (load balancing overhead)
    network_latency = options.NETWORK_LATENCY_BASE + (options.NETWORK_LATENCY_SCALE * math.log(1 + replicas))
    
    # Coordination overhead for distributed systems
    coordination_latency = options.COORDINATION_LATENCY_SCALE * math.log(1 + replicas)
    
    latency = (processing_latency + batch_wait_time + queue_latency + 
               network_latency + coordination_latency)
    
    # --- MEMORY MODEL ---
    # Memory scales with replicas and batch size
    batch_memory_multiplier = 1.0 + (batch_size - 1) * options.BATCH_MEMORY_INCREMENT
    
    # Concurrency requires buffer memory for queued requests
    concurrency_memory = options.CONCURRENCY_MEMORY_PER_REQUEST * concurrency
    
    # Memory overhead for coordination in larger clusters
    cluster_overhead = options.CLUSTER_MEMORY_OVERHEAD_SCALE * math.log(1 + replicas)
    
    max_memory_usage = (options.BASE_MEMORY_PER_REPLICA * replicas * batch_memory_multiplier + 
                        concurrency_memory + cluster_overhead)
    
    # Return results
    results = {
        'throughput': throughput,
        'latency': latency,
        'utilization': utilization,
        'cpu_memory_per_replica': max_memory_usage / replicas if replicas > 0 else 0,
        'gpu_memory_per_replica': batch_size * options.MODEL_MEMORY_PER_REQUEST + options.MODEL_MEMORY_OVERHEAD,
    }
    
    return results



def apply_categorical_settings(
    base_options: PerformanceOptions,
    deployment: Optional[int] = None,
    scheduler: Optional[int] = None,
    network_topology: Optional[int] = None,
    load_balancer: Optional[int] = None,
    batching_mode: Optional[int] = None,
    autoscaling: Optional[int] = None,
    hardware: Optional[int] = None,
    framework: Optional[int] = None,
    cache: Optional[int] = None
) -> PerformanceOptions:
    """
    Apply categorical settings to a base PerformanceOptions instance using delta-based updates.

    This function applies changes compositionally:
    - MULTIPLY operations are applied multiplicatively
    - ADD operations are applied additively
    - SET operations override (used sparingly)

    Args:
        base_options: Base PerformanceOptions to start from
        deployment: DEPLOYMENT setting (0, 1, or 2)
        scheduler: SCHEDULER setting (0, 1, or 2)
        network_topology: NETWORK_TOPOLOGY setting (0, 1, or 2)
        load_balancer: LOAD_BALANCER_STRATEGY setting (0, 1, or 2)
        batching_mode: BATCHING_MODE setting (0, 1, or 2)
        autoscaling: AUTOSCALING_POLICY setting (0, 1, or 2)
        hardware: HARDWARE_TIER setting (0, 1, or 2)
        framework: MODEL_SERVING_FRAMEWORK setting (0, 1, or 2)
        cache: CACHE_STRATEGY setting (0, 1, or 2)

    Returns:
        New PerformanceOptions instance with applied settings
    """
    # Convert base_options to dict
    options_dict = asdict(base_options)

    # Collect all updates to apply
    settings_to_apply = []
    if deployment is not None:
        settings_to_apply.append(DEPLOYMENT[deployment])
    if scheduler is not None:
        settings_to_apply.append(SCHEDULER[scheduler])
    if network_topology is not None:
        settings_to_apply.append(NETWORK_TOPOLOGY[network_topology])
    if load_balancer is not None:
        settings_to_apply.append(LOAD_BALANCER_STRATEGY[load_balancer])
    if batching_mode is not None:
        settings_to_apply.append(BATCHING_MODE[batching_mode])
    if autoscaling is not None:
        settings_to_apply.append(AUTOSCALING_POLICY[autoscaling])
    if hardware is not None:
        settings_to_apply.append(HARDWARE_TIER[hardware])
    if framework is not None:
        settings_to_apply.append(MODEL_SERVING_FRAMEWORK[framework])
    if cache is not None:
        settings_to_apply.append(CACHE_STRATEGY[cache])
    
    # Apply updates in order, respecting the update mode
    for setting_config in settings_to_apply:
        for param_name, (value, mode) in setting_config.items():
            if param_name not in options_dict:
                continue
                
            base_value = options_dict[param_name]
            
            if mode == UpdateMode.MULTIPLY:
                options_dict[param_name] = base_value * value
            elif mode == UpdateMode.ADD:
                options_dict[param_name] = base_value + value
            elif mode == UpdateMode.SET:
                options_dict[param_name] = value
            else:
                raise ValueError(f"Unknown UpdateMode: {mode}")
    
    # Create and return new PerformanceOptions instance
    return PerformanceOptions(**options_dict)


# Example usage
if __name__ == "__main__":
    # Test apply_categorical_settings
    base = PerformanceOptions()
    
    # Apply multiple categorical settings
    configured = apply_categorical_settings(
        base,
    #    network_topology=1,
    #    hardware=2,
    #    framework=0,
    #    cache=2,
    )

    # Test various configurations with default options
    test_configs = [
        {'replicas': 1, 'concurrency': 1, 'batch_size': 16},
        #{'replicas': 1, 'concurrency': 10, 'batch_size': 16},
        #{'replicas': 1, 'concurrency': 100, 'batch_size': 16},
        #{'replicas': 1, 'concurrency': 1000, 'batch_size': 16},
        #{'replicas': 4, 'concurrency': 40, 'batch_size': 8},
        #{'replicas': 10, 'concurrency': 200, 'batch_size': 16},
        #{'replicas': 20, 'concurrency': 500, 'batch_size': 32},
        #{'replicas': 50, 'concurrency': 500, 'batch_size': 32},
        {'replicas': 100, 'concurrency': 5000, 'batch_size': 32},
        {'replicas': 200, 'concurrency': 5000, 'batch_size': 32},
        {'replicas': 500, 'concurrency': 5000, 'batch_size': 32},
        {'replicas': 1000, 'concurrency': 5000, 'batch_size': 32},
        {'replicas': 2000, 'concurrency': 5000, 'batch_size': 32},
        {'replicas': 5000, 'concurrency': 5000, 'batch_size': 32},
    ]
    
    print("Server Cluster Performance Simulation Results:")
    print("=" * 70)
    
    for config in test_configs:
        results = simulate_performance(config, configured)
        print(f"\nConfig: {config}")
        pprint(results)
        print()

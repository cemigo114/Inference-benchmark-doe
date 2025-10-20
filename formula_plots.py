"""
Visualization of formulas used in modelsim_v2.py simulate_performance function.
Each plot function corresponds to a formula or stanza in the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from modelsim_v2 import PerformanceOptions


def plot_formula_1_max_batch_size():
    """Plot: max_batch_size_per_replica vs MAX_MEMORY_PER_REPLICA"""
    options = PerformanceOptions()

    # Vary MAX_MEMORY_PER_REPLICA
    max_memory_values = np.linspace(5, 100, 100)
    max_batch_sizes = []

    for max_mem in max_memory_values:
        max_batch_size_per_replica = max(0, math.floor(
            (max_mem - options.MODEL_MEMORY_OVERHEAD) / options.MODEL_MEMORY_PER_REQUEST
        ))
        max_batch_sizes.append(max_batch_size_per_replica)

    plt.figure(figsize=(10, 6))
    plt.plot(max_memory_values, max_batch_sizes, linewidth=2)
    plt.xlabel('MAX_MEMORY_PER_REPLICA (GB)')
    plt.ylabel('max_batch_size_per_replica')
    plt.title('GPU Memory Constraint: Maximum Batch Size per Replica')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('formula_1_max_batch_size.png', dpi=150)
    plt.close()
    print("Saved: formula_1_max_batch_size.png")


def plot_formula_2_replica_efficiency():
    """Plot: replica_efficiency vs replicas"""
    options = PerformanceOptions()

    # Vary replicas (log scale makes sense here)
    replicas_values = np.unique(np.round(2 ** np.linspace(0, 10, 100)))
    replica_efficiency_values = []

    for replicas in replicas_values:
        replica_efficiency = 1.0 - (options.COORDINATION_OVERHEAD * math.log(1 + replicas))
        replica_efficiency_values.append(replica_efficiency)

    plt.figure(figsize=(10, 6))
    plt.plot(replicas_values, replica_efficiency_values, linewidth=2)
    plt.xlabel('replicas')
    plt.ylabel('replica_efficiency')
    plt.title('Replica Efficiency: Diminishing Returns from Coordination Overhead')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('formula_2_replica_efficiency.png', dpi=150)
    plt.close()
    print("Saved: formula_2_replica_efficiency.png")


def plot_formula_3_batch_efficiency():
    """Plot: batch_efficiency vs batch_size"""
    options = PerformanceOptions()

    # Vary batch_size
    batch_size_values = np.arange(1, 20, 0.1)
    batch_efficiency_values = []

    for batch_size in batch_size_values:
        batch_efficiency = min(batch_size, options.BATCH_EFFICIENCY_THRESHOLD)
        batch_efficiency_values.append(batch_efficiency)

    plt.figure(figsize=(10, 6))
    plt.plot(batch_size_values, batch_efficiency_values, linewidth=2)
    plt.axvline(x=options.BATCH_EFFICIENCY_THRESHOLD, color='r', linestyle='--',
                label=f'Threshold = {options.BATCH_EFFICIENCY_THRESHOLD}')
    plt.xlabel('batch_size')
    plt.ylabel('batch_efficiency')
    plt.title('Batch Efficiency: Linear Scaling Until Threshold, Then Plateau')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('formula_3_batch_efficiency.png', dpi=150)
    plt.close()
    print("Saved: formula_3_batch_efficiency.png")


def plot_formula_4_concurrency_load_factor():
    """Plot: concurrency_load_factor vs concurrency (for fixed replicas)"""
    options = PerformanceOptions()
    replicas = 10  # Fixed for this plot

    # Vary concurrency
    concurrency_values = np.linspace(1, 200, 100)
    load_factor_values = []

    for concurrency in concurrency_values:
        concurrency_load_factor = concurrency / (replicas * options.OPTIMAL_CONCURRENCY_PER_REPLICA)
        load_factor_values.append(concurrency_load_factor)

    plt.figure(figsize=(10, 6))
    plt.plot(concurrency_values, load_factor_values, linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Capacity Threshold')
    plt.xlabel('concurrency')
    plt.ylabel('concurrency_load_factor')
    plt.title(f'Concurrency Load Factor (replicas={replicas})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('formula_4_concurrency_load_factor.png', dpi=150)
    plt.close()
    print("Saved: formula_4_concurrency_load_factor.png")


def plot_formula_5_concurrency_efficiency():
    """Plot: concurrency_efficiency vs concurrency_load_factor"""
    options = PerformanceOptions()

    # Vary concurrency_load_factor
    load_factor_values = np.linspace(0.01, 3.0, 200)
    efficiency_values = []

    for concurrency_load_factor in load_factor_values:
        if concurrency_load_factor <= 1.0:
            # Under capacity: near-linear scaling
            concurrency_efficiency = (options.UNDER_CAPACITY_BASE_EFFICIENCY +
                                      options.UNDER_CAPACITY_SCALE_FACTOR * concurrency_load_factor)
        else:
            # Over capacity: throughput plateaus and degrades
            concurrency_efficiency = 1.0 / (1.0 + options.OVER_CAPACITY_DEGRADATION * (concurrency_load_factor - 1.0))
        efficiency_values.append(concurrency_efficiency)

    plt.figure(figsize=(10, 6))
    plt.plot(load_factor_values, efficiency_values, linewidth=2)
    plt.axvline(x=1.0, color='r', linestyle='--', label='Capacity Threshold')
    plt.xlabel('concurrency_load_factor')
    plt.ylabel('concurrency_efficiency')
    plt.title('Concurrency Efficiency: Linear Under Capacity, Degradation Over Capacity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('formula_5_concurrency_efficiency.png', dpi=150)
    plt.close()
    print("Saved: formula_5_concurrency_efficiency.png")


def plot_formula_6_throughput():
    """Plot: throughput vs replicas (with fixed concurrency and batch_size)"""
    options = PerformanceOptions()
    concurrency = 100  # Fixed
    batch_size = 8     # Fixed

    # Vary replicas
    replicas_values = np.unique(np.round(2 ** np.linspace(0, 8, 50)))
    throughput_values = []

    for replicas in replicas_values:
        replica_efficiency = 1.0 - (options.COORDINATION_OVERHEAD * math.log(1 + replicas))
        batch_efficiency = min(batch_size, options.BATCH_EFFICIENCY_THRESHOLD)
        concurrency_load_factor = concurrency / (replicas * options.OPTIMAL_CONCURRENCY_PER_REPLICA)

        if concurrency_load_factor <= 1.0:
            concurrency_efficiency = (options.UNDER_CAPACITY_BASE_EFFICIENCY +
                                      options.UNDER_CAPACITY_SCALE_FACTOR * concurrency_load_factor)
        else:
            concurrency_efficiency = 1.0 / (1.0 + options.OVER_CAPACITY_DEGRADATION * (concurrency_load_factor - 1.0))

        throughput = (options.BASE_THROUGHPUT_PER_REPLICA * replicas * replica_efficiency *
                      batch_efficiency * concurrency_efficiency)
        throughput_values.append(throughput)

    plt.figure(figsize=(10, 6))
    plt.plot(replicas_values, throughput_values, linewidth=2)
    plt.xlabel('replicas')
    plt.ylabel('throughput (req/s)')
    plt.title(f'Total Throughput vs Replicas (concurrency={concurrency}, batch_size={batch_size})')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('formula_6_throughput.png', dpi=150)
    plt.close()
    print("Saved: formula_6_throughput.png")


def plot_formula_7_batch_wait_time():
    """Plot: batch_wait_time vs batch_size"""
    options = PerformanceOptions()

    # Vary batch_size
    batch_size_values = np.arange(1, 33)
    batch_wait_time_values = []

    for batch_size in batch_size_values:
        batch_wait_time = (batch_size - 1) * options.BATCH_WAIT_TIME_PER_ITEM
        batch_wait_time_values.append(batch_wait_time)

    plt.figure(figsize=(10, 6))
    plt.plot(batch_size_values, batch_wait_time_values, linewidth=2)
    plt.xlabel('batch_size')
    plt.ylabel('batch_wait_time (ms)')
    plt.title('Batch Wait Time: Linear Increase with Batch Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('formula_7_batch_wait_time.png', dpi=150)
    plt.close()
    print("Saved: formula_7_batch_wait_time.png")


def plot_formula_8_queue_latency():
    """Plot: queue_latency vs utilization"""
    options = PerformanceOptions()

    # Vary utilization
    utilization_values = np.linspace(0.01, 2.0, 300)
    queue_latency_values = []

    for utilization in utilization_values:
        if utilization < options.QUEUE_UTILIZATION_THRESHOLD:
            queue_latency = options.QUEUE_LATENCY_BASE * utilization / (1.0 - utilization)
        else:
            # Heavy load: exponential growth in queue time
            excess_utilization = utilization - options.QUEUE_UTILIZATION_THRESHOLD
            queue_latency = (options.QUEUE_LATENCY_BASE + options.QUEUE_LATENCY_HEAVY_LOAD *
                                math.exp(min(options.QUEUE_EXPONENTIAL_FACTOR * excess_utilization, 100)))
        queue_latency_values.append(queue_latency)

    plt.figure(figsize=(10, 6))
    plt.plot(utilization_values, queue_latency_values, linewidth=2)
    plt.axvline(x=options.QUEUE_UTILIZATION_THRESHOLD, color='r', linestyle='--',
                label=f'Threshold = {options.QUEUE_UTILIZATION_THRESHOLD}')
    plt.axvline(x=1.0, color='orange', linestyle='--', alpha=0.5, label='Full Capacity')
    plt.xlabel('utilization')
    plt.ylabel('queue_latency (ms)')
    plt.title('Queue Latency: M/M/c Model with Exponential Growth Under Heavy Load')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('formula_8_queue_latency.png', dpi=150)
    plt.close()
    print("Saved: formula_8_queue_latency.png")


def plot_formula_9_network_latency():
    """Plot: network_latency vs replicas"""
    options = PerformanceOptions()

    # Vary replicas
    replicas_values = np.unique(np.round(2 ** np.linspace(0, 10, 100)))
    network_latency_values = []

    for replicas in replicas_values:
        network_latency = options.NETWORK_LATENCY_BASE + (options.NETWORK_LATENCY_SCALE * math.log(1 + replicas))
        network_latency_values.append(network_latency)

    plt.figure(figsize=(10, 6))
    plt.plot(replicas_values, network_latency_values, linewidth=2)
    plt.xlabel('replicas')
    plt.ylabel('network_latency (ms)')
    plt.title('Network Latency: Logarithmic Growth with Cluster Size')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('formula_9_network_latency.png', dpi=150)
    plt.close()
    print("Saved: formula_9_network_latency.png")


def plot_formula_10_coordination_latency():
    """Plot: coordination_latency vs replicas"""
    options = PerformanceOptions()

    # Vary replicas
    replicas_values = np.unique(np.round(2 ** np.linspace(0, 10, 100)))
    coordination_latency_values = []

    for replicas in replicas_values:
        coordination_latency = options.COORDINATION_LATENCY_SCALE * math.log(1 + replicas)
        coordination_latency_values.append(coordination_latency)

    plt.figure(figsize=(10, 6))
    plt.plot(replicas_values, coordination_latency_values, linewidth=2)
    plt.xlabel('replicas')
    plt.ylabel('coordination_latency (ms)')
    plt.title('Coordination Latency: Logarithmic Growth with Cluster Size')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('formula_10_coordination_latency.png', dpi=150)
    plt.close()
    print("Saved: formula_10_coordination_latency.png")


def plot_formula_11_total_latency():
    """Plot: total latency components vs batch_size (fixed replicas, concurrency)"""
    options = PerformanceOptions()
    replicas = 10
    concurrency = 100

    # Vary batch_size
    batch_size_values = np.arange(1, 33)
    processing_latency_values = []
    batch_wait_time_values = []
    queue_latency_values = []
    network_latency_values = []
    coordination_latency_values = []
    total_latency_values = []

    for batch_size in batch_size_values:
        processing_latency = options.BASE_LATENCY
        batch_wait_time = (batch_size - 1) * options.BATCH_WAIT_TIME_PER_ITEM

        concurrency_load_factor = concurrency / (replicas * options.OPTIMAL_CONCURRENCY_PER_REPLICA)
        utilization = concurrency_load_factor

        if utilization < options.QUEUE_UTILIZATION_THRESHOLD:
            queue_latency = options.QUEUE_LATENCY_BASE * utilization / (1.0 - utilization)
        else:
            excess_utilization = utilization - options.QUEUE_UTILIZATION_THRESHOLD
            queue_latency = (options.QUEUE_LATENCY_BASE + options.QUEUE_LATENCY_HEAVY_LOAD *
                                math.exp(min(options.QUEUE_EXPONENTIAL_FACTOR * excess_utilization, 100)))

        network_latency = options.NETWORK_LATENCY_BASE + (options.NETWORK_LATENCY_SCALE * math.log(1 + replicas))
        coordination_latency = options.COORDINATION_LATENCY_SCALE * math.log(1 + replicas)

        total_latency = (processing_latency + batch_wait_time + queue_latency +
                        network_latency + coordination_latency)

        processing_latency_values.append(processing_latency)
        batch_wait_time_values.append(batch_wait_time)
        queue_latency_values.append(queue_latency)
        network_latency_values.append(network_latency)
        coordination_latency_values.append(coordination_latency)
        total_latency_values.append(total_latency)

    plt.figure(figsize=(10, 6))
    plt.plot(batch_size_values, processing_latency_values, label='Processing', linewidth=2)
    plt.plot(batch_size_values, batch_wait_time_values, label='Batch Wait', linewidth=2)
    plt.plot(batch_size_values, queue_latency_values, label='Queue', linewidth=2)
    plt.plot(batch_size_values, network_latency_values, label='Network', linewidth=2)
    plt.plot(batch_size_values, coordination_latency_values, label='Coordination', linewidth=2)
    plt.plot(batch_size_values, total_latency_values, label='Total', linewidth=2, linestyle='--', color='black')
    plt.xlabel('batch_size')
    plt.ylabel('latency (ms)')
    plt.title(f'Latency Components (replicas={replicas}, concurrency={concurrency})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('formula_11_total_latency.png', dpi=150)
    plt.close()
    print("Saved: formula_11_total_latency.png")


def plot_formula_12_batch_memory_multiplier():
    """Plot: batch_memory_multiplier vs batch_size"""
    options = PerformanceOptions()

    # Vary batch_size
    batch_size_values = np.arange(1, 33)
    batch_memory_multiplier_values = []

    for batch_size in batch_size_values:
        batch_memory_multiplier = 1.0 + (batch_size - 1) * options.BATCH_MEMORY_INCREMENT
        batch_memory_multiplier_values.append(batch_memory_multiplier)

    plt.figure(figsize=(10, 6))
    plt.plot(batch_size_values, batch_memory_multiplier_values, linewidth=2)
    plt.xlabel('batch_size')
    plt.ylabel('batch_memory_multiplier')
    plt.title('Batch Memory Multiplier: Linear Increase with Batch Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('formula_12_batch_memory_multiplier.png', dpi=150)
    plt.close()
    print("Saved: formula_12_batch_memory_multiplier.png")


def plot_formula_13_concurrency_memory():
    """Plot: concurrency_memory vs concurrency"""
    options = PerformanceOptions()

    # Vary concurrency
    concurrency_values = np.linspace(1, 1000, 100)
    concurrency_memory_values = []

    for concurrency in concurrency_values:
        concurrency_memory = options.CONCURRENCY_MEMORY_PER_REQUEST * concurrency
        concurrency_memory_values.append(concurrency_memory)

    plt.figure(figsize=(10, 6))
    plt.plot(concurrency_values, concurrency_memory_values, linewidth=2)
    plt.xlabel('concurrency')
    plt.ylabel('concurrency_memory (GB)')
    plt.title('Concurrency Memory: Linear with Number of Concurrent Requests')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('formula_13_concurrency_memory.png', dpi=150)
    plt.close()
    print("Saved: formula_13_concurrency_memory.png")


def plot_formula_14_cluster_overhead():
    """Plot: cluster_overhead vs replicas"""
    options = PerformanceOptions()

    # Vary replicas
    replicas_values = np.unique(np.round(2 ** np.linspace(0, 10, 100)))
    cluster_overhead_values = []

    for replicas in replicas_values:
        cluster_overhead = options.CLUSTER_MEMORY_OVERHEAD_SCALE * math.log(1 + replicas)
        cluster_overhead_values.append(cluster_overhead)

    plt.figure(figsize=(10, 6))
    plt.plot(replicas_values, cluster_overhead_values, linewidth=2)
    plt.xlabel('replicas')
    plt.ylabel('cluster_overhead (GB)')
    plt.title('Cluster Memory Overhead: Logarithmic Growth with Cluster Size')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('formula_14_cluster_overhead.png', dpi=150)
    plt.close()
    print("Saved: formula_14_cluster_overhead.png")


def plot_formula_15_total_memory():
    """Plot: total memory components vs replicas (fixed batch_size, concurrency)"""
    options = PerformanceOptions()
    batch_size = 16
    concurrency = 200

    # Vary replicas
    replicas_values = np.unique(np.round(2 ** np.linspace(0, 8, 50)))
    replica_memory_values = []
    concurrency_memory_values = []
    cluster_overhead_values = []
    total_memory_values = []

    for replicas in replicas_values:
        batch_memory_multiplier = 1.0 + (batch_size - 1) * options.BATCH_MEMORY_INCREMENT
        replica_memory = options.BASE_MEMORY_PER_REPLICA * replicas * batch_memory_multiplier
        concurrency_memory = options.CONCURRENCY_MEMORY_PER_REQUEST * concurrency
        cluster_overhead = options.CLUSTER_MEMORY_OVERHEAD_SCALE * math.log(1 + replicas)

        max_memory_usage = replica_memory + concurrency_memory + cluster_overhead

        replica_memory_values.append(replica_memory)
        concurrency_memory_values.append(concurrency_memory)
        cluster_overhead_values.append(cluster_overhead)
        total_memory_values.append(max_memory_usage)

    plt.figure(figsize=(10, 6))
    plt.plot(replicas_values, replica_memory_values, label='Replica Memory', linewidth=2)
    plt.plot(replicas_values, concurrency_memory_values, label='Concurrency Memory', linewidth=2)
    plt.plot(replicas_values, cluster_overhead_values, label='Cluster Overhead', linewidth=2)
    plt.plot(replicas_values, total_memory_values, label='Total Memory', linewidth=2,
             linestyle='--', color='black')
    plt.xlabel('replicas')
    plt.ylabel('memory (GB)')
    plt.title(f'Memory Components (batch_size={batch_size}, concurrency={concurrency})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('formula_15_total_memory.png', dpi=150)
    plt.close()
    print("Saved: formula_15_total_memory.png")


def main():
    """Generate all formula plots"""
    print("Generating formula plots from modelsim_v2.py simulate_performance()...")
    print()

    plot_formula_1_max_batch_size()
    plot_formula_2_replica_efficiency()
    plot_formula_3_batch_efficiency()
    plot_formula_4_concurrency_load_factor()
    plot_formula_5_concurrency_efficiency()
    plot_formula_6_throughput()
    plot_formula_7_batch_wait_time()
    plot_formula_8_queue_latency()
    plot_formula_9_network_latency()
    plot_formula_10_coordination_latency()
    plot_formula_11_total_latency()
    plot_formula_12_batch_memory_multiplier()
    plot_formula_13_concurrency_memory()
    plot_formula_14_cluster_overhead()
    plot_formula_15_total_memory()

    print()
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()

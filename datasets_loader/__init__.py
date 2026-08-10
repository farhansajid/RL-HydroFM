"""
Dataset Loaders and Benchmarks for Water Resources & Remote Sensing.
"""
from datasets_loader.water_benchmarks import (
    load_water_benchmark,
    get_benchmark_classes,
    generate_synthetic_benchmark_if_needed,
    WATER_BENCHMARK_CONFIGS,
)

__all__ = [
    'load_water_benchmark',
    'get_benchmark_classes',
    'generate_synthetic_benchmark_if_needed',
    'WATER_BENCHMARK_CONFIGS',
]

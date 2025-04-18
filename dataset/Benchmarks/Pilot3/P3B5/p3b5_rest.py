import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "learning_rate_min", "action": "store", "type": float},
    {"name": "log_interval", "action": "store", "type": int},
    {"name": "weight_decay", "action": "store", "type": float},
    {"name": "grad_clip", "action": "store", "type": int},
    {"name": "unrolled", "action": "store", "type": candle.str2bool},
]

required = [
    "learning_rate",
    "learning_rate_min",
    "momentum",
    "weight_decay",
    "grad_clip",
    "rng_seed",
    "unrolled",
    "batch_size",
    "epochs",
]


class BenchmarkP3B5(candle.Benchmark):
    """Benchmark for P3B5"""

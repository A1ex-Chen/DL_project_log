import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "grad_clip", "type": int},
    {"name": "learning_rate_min", "type": float, "help": "Minimum learning rate"},
    {"name": "log_interval", "type": int, "help": "Logging interval"},
    {"name": "unrolled", "type": candle.str2bool},
    {"name": "weight_decay", "type": float},
    {"name": "grad_clip", "type": int},
]

REQUIRED = [
    "learning_rate",
    "learning_rate_min",
    "momentum",
    "weight_decay",
    "grad_clip",
    "rng_seed",
    "batch_size",
    "epochs",
]


class AdvancedExample(candle.Benchmark):
    """Example for Advanced use of DARTS"""

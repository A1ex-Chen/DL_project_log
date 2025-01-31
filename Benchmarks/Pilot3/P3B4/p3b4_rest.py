from __future__ import print_function

import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "attention_heads", "action": "store", "type": int},
    {"name": "attention_size", "action": "store", "type": int},
    {"name": "embed_train", "action": "store", "type": candle.str2bool},
    {"name": "min_lines", "action": "store", "type": int},
    {"name": "max_lines", "action": "store", "type": int},
    {"name": "min_words", "action": "store", "type": int},
    {"name": "max_words", "action": "store", "type": int},
    {"name": "wv_len", "action": "store", "type": int},
]

required = [
    "learning_rate",
    "batch_size",
    "epochs",
    "dropout",
    "optimizer",
    "wv_len",
    "min_lines",
    "max_lines",
    "min_words",
    "max_words",
    "attention_size",
    "embed_train",
    "attention_heads",
]


class BenchmarkP3B4(candle.Benchmark):
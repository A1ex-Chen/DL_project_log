from __future__ import print_function

import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {
        "name": "rnn_size",
        "action": "store",
        "type": int,
        "help": "size of LSTM internal state",
    },
    {"name": "n_layers", "action": "store", "help": "number of layers in the LSTM"},
    {"name": "do_sample", "type": candle.str2bool, "help": "generate synthesized text"},
    {
        "name": "temperature",
        "action": "store",
        "type": float,
        "help": "variability of text synthesis",
    },
    {"name": "primetext", "action": "store", "help": "seed string of text synthesis"},
    {
        "name": "length",
        "action": "store",
        "type": int,
        "help": "length of synthesized text",
    },
]

required = [
    "train_data",
    "rnn_size",
    "epochs",
    "n_layers",
    "learning_rate",
    "dropout",
    "recurrent_dropout",
    "temperature",
    "primetext",
    "length",
]


class BenchmarkP3B2(candle.Benchmark):
import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "classes", "type": int, "default": 2},
    {"name": "label_noise", "type": float},
    {"name": "std_dev", "type": float},
    {"name": "feature_col", "type": int},
    {"name": "sample_ids", "type": int},
    {"name": "feature_threshold", "type": float},
    {"name": "add_noise", "type": candle.str2bool},
    {"name": "noise_correlated", "type": candle.str2bool},
    {"name": "noise_column", "type": candle.str2bool},
    {"name": "noise_cluster", "type": candle.str2bool},
    {"name": "noise_gaussian", "type": candle.str2bool},
    {"name": "noise_type", "type": str},
]

required = [
    "data_url",
    "train_data",
    "test_data",
    "model_name",
    "conv",
    "dense",
    "activation",
    "out_activation",
    "loss",
    "optimizer",
    "metrics",
    "epochs",
    "batch_size",
    "learning_rate",
    "dropout",
    "classes",
    "pool",
    "output_dir",
    "timeout",
]


class BenchmarkNT3(candle.Benchmark):
            # print(self.additional_definitions)
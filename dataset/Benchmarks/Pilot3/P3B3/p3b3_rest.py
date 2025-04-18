from __future__ import print_function

import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {
        "name": "train_features",
        "action": "store",
        "default": "data/task0_0_train_feature.csv;data/task1_0_train_feature.csv;data/task2_0_train_feature.csv",
        "help": "training feature data filenames",
    },
    {
        "name": "train_truths",
        "action": "store",
        "default": "data/task0_0_train_label.csv;data/task1_0_train_label.csv;data/task2_0_train_label.csv",
        "help": "training truth data filenames",
    },
    {
        "name": "valid_features",
        "action": "store",
        "default": "data/task0_0_test_feature.csv;data/task1_0_test_feature.csv;data/task2_0_test_feature.csv",
        "help": "validation feature data filenames",
    },
    {
        "name": "valid_truths",
        "action": "store",
        "default": "data/task0_0_test_label.csv;data/task1_0_test_label.csv;data/task2_0_test_label.csv",
        "help": "validation truth data filenames",
    },
    {
        "name": "output_files",
        "action": "store",
        "default": "result0_0.csv;result1_0.csv;result2_0.csv",
        "help": "output filename",
    },
    {
        "name": "shared_nnet_spec",
        "nargs": "+",
        "type": int,
        "help": "network structure of shared layer",
    },
    {
        "name": "ind_nnet_spec",
        "action": "list-of-lists",
        "help": "network structure of task-specific layer",
    },
    {
        "name": "case",
        "default": "CenterZ",
        "choices": ["Full", "Center", "CenterZ"],
        "help": "case classes",
    },
    {
        "name": "fig",
        "type": candle.str2bool,
        "default": False,
        "help": "Generate Prediction Figure",
    },
    {"name": "feature_names", "nargs": "+", "type": str},
    {"name": "n_fold", "action": "store", "type": int},
    {"name": "emb_l2", "action": "store", "type": float},
    {"name": "w_l2", "action": "store", "type": float},
    {"name": "wv_len", "action": "store", "type": int},
    {"name": "filter_sets", "nargs": "+", "type": int},
    {"name": "filter_sizes", "nargs": "+", "type": int},
    {"name": "num_filters", "nargs": "+", "type": int},
    {
        "name": "task_list",
        "nargs": "+",
        "type": int,
        "help": "list of task indices to use",
    },
    {
        "name": "task_names",
        "nargs": "+",
        "type": int,
        "help": "list of names corresponding to each task to use",
    },
]


required = [
    "learning_rate",
    "batch_size",
    "epochs",
    "dropout",
    "optimizer",
    "wv_len",
    "filter_sizes",
    "filter_sets",
    "num_filters",
    "emb_l2",
    "w_l2",
]


class BenchmarkP3B3(candle.Benchmark):
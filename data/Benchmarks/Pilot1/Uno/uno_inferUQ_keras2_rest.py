#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import logging
import os
from itertools import cycle

import candle
import numpy as np
import pandas as pd
import uno as benchmark
from tensorflow import keras
from tensorflow.keras import backend as K
from uno_baseline_keras2 import evaluate_prediction
from uno_data import CombinedDataGenerator, CombinedDataLoader, read_IDs_file
from uno_trainUQ_keras2 import extension_from_parameters, log_evaluation
from unoUQ_data import FromFileDataGenerator

logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


additional_definitions_local = [
    {
        "name": "uq_infer_file",
        "default": argparse.SUPPRESS,
        "action": "store",
        "help": "File to do inference",
    },
    {
        "name": "uq_infer_given_drugs",
        "type": candle.str2bool,
        "default": False,
        "help": "Use given inference file to obtain drug ids to do inference",
    },
    {
        "name": "uq_infer_given_cells",
        "type": candle.str2bool,
        "default": False,
        "help": "Use given inference file to obtain cell ids to do inference",
    },
    {
        "name": "uq_infer_given_indices",
        "type": candle.str2bool,
        "default": False,
        "help": "Use given inference file to obtain indices to do inference",
    },
    {
        "name": "model_file",
        "type": str,
        "default": "saved.model.h5",
        "help": "trained model file",
    },
    {
        "name": "weights_file",
        "type": str,
        "default": "saved.weights.h5",
        "help": "trained weights file (loading model file alone sometimes does not work in keras)",
    },
    {
        "name": "n_pred",
        "type": int,
        "default": 1,
        "help": "the number of predictions to make for each sample-drug combination for uncertainty quantification",
    },
]

required_local = (
    "model_file",
    "weights_file",
    "uq_infer_file",
    "agg_dose",
    "batch_size",
)
















if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
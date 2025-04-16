#! /usr/bin/env python

from __future__ import division, print_function

import logging
import os

import candle
import uno as benchmark
from tensorflow.keras import backend as K
from uno_data import CombinedDataLoader
from uno_trainUQ_keras2 import extension_from_parameters

logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"








if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
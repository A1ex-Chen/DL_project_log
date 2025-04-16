from __future__ import print_function

import os

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    LocallyConnected1D,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential, model_from_json

file_path = os.path.dirname(os.path.realpath(__file__))

import candle
import tc1 as bmk








if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:  # theano does not have this function
        pass
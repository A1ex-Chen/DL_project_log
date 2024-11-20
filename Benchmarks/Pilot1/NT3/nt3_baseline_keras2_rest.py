from __future__ import print_function

import os

import candle
import nt3 as bmk
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
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
from tensorflow.keras.utils import to_categorical










if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:  # theano does not have this function
        pass
from __future__ import absolute_import, print_function

import os

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2DTranspose,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

file_path = os.path.dirname(os.path.realpath(__file__))








'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

keras_model.py: CIFAR10_ResNetv1 from eembc
'''

import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

#get model


#define model
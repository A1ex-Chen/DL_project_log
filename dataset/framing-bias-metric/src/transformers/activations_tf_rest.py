import math

import tensorflow as tf










ACT2FN = {
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.activations.swish,
    "silu": tf.keras.activations.swish,
    "gelu_new": tf.keras.layers.Activation(gelu_new),
    "mish": tf.keras.layers.Activation(mish),
    "tanh": tf.keras.activations.tanh,
    "gelu_fast": tf.keras.layers.Activation(gelu_fast),
}


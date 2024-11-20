import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.python.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf


# VectorQuantizer layer
class VectorQuantizer(layers.Layer):




class ResidualLayer(layers.Layer):



class Encoder(layers.Layer):



class Decoder(layers.Layer):



class VQVAE(keras.models.Model):



if __name__ == "__main__":
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    vqvae = VQVAE(K=128, D=256)
    vqvae.compile(optimizer=keras.optimizers.Adam())
    vqvae.fit(x_train, epochs=2, batch_size=64)




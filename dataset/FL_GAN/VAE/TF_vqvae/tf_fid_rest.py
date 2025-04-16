import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
import tensorflow_gan as tfgan
from tf_utils import load_data

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

session = tf.compat.v1.InteractiveSession()
# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64

# Run images through Inception.
inception_images = tf.compat.v1.placeholder(tf.float32, [None, None, None, None], name='inception_images')
activations1 = tf.compat.v1.placeholder(tf.float32, [None, None], name='activations1')
activations2 = tf.compat.v1.placeholder(tf.float32, [None, None], name='activations2')
fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_FINAL_POOL = 'pool_3'




activations = inception_activations()








if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_data("mnist")

    fid = get_fid(train_data[:10], test_data[:10])
    print(fid)
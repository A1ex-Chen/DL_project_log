# Lint as: python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common modules for callbacks."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os
from typing import Any, List, MutableMapping, Text
import tensorflow as tf
from tensorflow import keras

from utils import optimizer_factory
##
#import horovod.tensorflow as hvd
import smdistributed.dataparallel.tensorflow as sdp
##
# Sagemaker uses s3 to save model, and since 2.6 s3 support needs tensorflow-io package
# https://giters.com/tensorflow/tensorflow/issues/51583?amp=1
from distutils.version import LooseVersion
if LooseVersion(tf.__version__) >= LooseVersion("2.6"):
    import tensorflow_io as tfio

import time






class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
  """A customized TensorBoard callback that tracks additional datapoints.

  Metrics tracked:
  - Global learning rate

  Attributes:
    log_dir: the path of the directory where to save the log files to be parsed
      by TensorBoard.
    track_lr: `bool`, whether or not to track the global learning rate.
    initial_step: the initial step, used for preemption recovery.
    **kwargs: Additional arguments for backwards compatibility. Possible key is
      `period`.
  """

  # TODO(b/146499062): track params, flops, log lr, l2 loss,
  # classification loss








class MovingAverageCallback(tf.keras.callbacks.Callback):
  """A Callback to be used with a `MovingAverage` optimizer.

  Applies moving average weights to the model during validation time to test
  and predict on the averaged weights rather than the current model weights.
  Once training is complete, the model weights will be overwritten with the
  averaged weights (by default).

  Attributes:
    overwrite_weights_on_train_end: Whether to overwrite the current model
      weights with the averaged weights from the moving average optimizer.
    **kwargs: Any additional callback arguments.
  """







class AverageModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  """Saves and, optionally, assigns the averaged weights.

  Taken from tfa.callbacks.AverageModelCheckpoint.

  Attributes:
    update_weights: If True, assign the moving average weights
      to the model, and save them. If False, keep the old
      non-averaged weights, but the saved model uses the
      average weights.
    See `tf.keras.callbacks.ModelCheckpoint` for the other args.
  """






class BatchTimestamp(object):
  """A structure to store batch time stamp."""





class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""


  @property

  @property

  @property







class EvalTimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""



  @property

  @property



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
"""Dataset utilities for vision tasks using TFDS and tf.data.Dataset."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os
from typing import Any, List, Optional, Tuple, Mapping, Union
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from utils import augment, preprocessing, Dali
##
#import horovod.tensorflow.keras as hvd
import smdistributed.dataparallel.tensorflow as sdp
##
import nvidia.dali.plugin.tf as dali_tf





AUGMENTERS = {
    'autoaugment': augment.AutoAugment,
    'randaugment': augment.RandAugment,
}

class Dataset:
  """An object for building datasets.

  Allows building various pipelines fetching examples, preprocessing, etc.
  Maintains additional state information calculated from the dataset, i.e.,
  training set split, batch size, and number of steps (batches).
  """




  @property

  @property

  @property

  @property

  @property

  @property

  @property

  @property


  # def augment_pipeline(self, image, label) -> Tuple[tf.Tensor, tf.Tensor]:
  #   image = self._augmenter.distort(image)
  #   return image, label






  @classmethod


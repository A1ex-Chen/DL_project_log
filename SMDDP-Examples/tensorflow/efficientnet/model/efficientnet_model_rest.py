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
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from typing import Any, Dict, Optional, List, Text, Tuple
import copy

import tensorflow as tf

from model.layers import simple_swish, hard_swish, identity, gelu, get_activation
from model.blocks import conv2d_block, mb_conv_block
from model.common_modules import round_filters, round_repeats, load_weights
from utils import preprocessing



# Config for a single MB Conv Block.
BlockConfig = {
  'input_filters': 0,
  'output_filters': 0,
  'kernel_size': 3,
  'num_repeat': 1,
  'expand_ratio': 1,
  'strides': (1, 1),
  'se_ratio': None,
  'id_skip': True,
  'fused_conv': False,
  'conv_type': 'depthwise'
  }

# Default Config for Efficientnet-B0.
ModelConfig = {
  'width_coefficient': 1.0,
  'depth_coefficient': 1.0,
  'resolution': 224,
  'dropout_rate': 0.2,
  'blocks': (
      # (input_filters, output_filters, kernel_size, num_repeat,
      #  expand_ratio, strides, se_ratio)
      # pylint: disable=bad-whitespace
      build_dict(name="BlockConfig", args=(32,  16,  3, 1, 1, (1, 1), 0.25)),
      build_dict(name="BlockConfig", args=(16,  24,  3, 2, 6, (2, 2), 0.25)),
      build_dict(name="BlockConfig", args=(24,  40,  5, 2, 6, (2, 2), 0.25)),
      build_dict(name="BlockConfig", args=(40,  80,  3, 3, 6, (2, 2), 0.25)),
      build_dict(name="BlockConfig", args=(80,  112, 5, 3, 6, (1, 1), 0.25)),
      build_dict(name="BlockConfig", args=(112, 192, 5, 4, 6, (2, 2), 0.25)),
      build_dict(name="BlockConfig", args=(192, 320, 3, 1, 6, (1, 1), 0.25)),
      # pylint: enable=bad-whitespace
  ),
  'stem_base_filters': 32,
  'top_base_filters': 1280,
  'activation': 'simple_swish',
  'batch_norm': 'default',
  'bn_momentum': 0.99,
  'bn_epsilon': 1e-3,
  # While the original implementation used a weight decay of 1e-5,
  # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
  'weight_decay': 5e-6,
  'drop_connect_rate': 0.2,
  'depth_divisor': 8,
  'min_depth': None,
  'use_se': True,
  'input_channels': 3,
  'num_classes': 1000,
  'model_name': 'efficientnet',
  'rescale_input': True,
  'data_format': 'channels_last',
  'dtype': 'float32',
  'weight_init': 'fan_in',
}

MODEL_CONFIGS = {
    # (width, depth, resolution, dropout)
    'efficientnet-b0': build_dict(name="ModelConfig", args=(1.0, 1.0, 224, 0.2)),
    'efficientnet-b1': build_dict(name="ModelConfig", args=(1.0, 1.1, 240, 0.2)),
    'efficientnet-b2': build_dict(name="ModelConfig", args=(1.1, 1.2, 260, 0.3)),
    'efficientnet-b3': build_dict(name="ModelConfig", args=(1.2, 1.4, 300, 0.3)),
    'efficientnet-b4': build_dict(name="ModelConfig", args=(1.4, 1.8, 380, 0.4)),
    'efficientnet-b5': build_dict(name="ModelConfig", args=(1.6, 2.2, 456, 0.4)),
    'efficientnet-b6': build_dict(name="ModelConfig", args=(1.8, 2.6, 528, 0.5)),
    'efficientnet-b7': build_dict(name="ModelConfig", args=(2.0, 3.1, 600, 0.5)),
    'efficientnet-b8': build_dict(name="ModelConfig", args=(2.2, 3.6, 672, 0.5)),
    'efficientnet-l2': build_dict(name="ModelConfig", args=(4.3, 5.3, 800, 0.5)),
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1 / 3.0,
        'mode': 'fan_in',
        'distribution': 'uniform'
    }
}




@tf.keras.utils.register_keras_serializable(package='Vision')
class EfficientNet(tf.keras.Model):
  """Wrapper class for an EfficientNet Keras model.

  Contains helper methods to build, manage, and save metadata about the model.
  """


  @classmethod
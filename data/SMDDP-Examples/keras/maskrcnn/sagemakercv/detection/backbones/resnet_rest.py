#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Resnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import variable_scope
from tensorflow.python.keras import backend

from ..builder import BACKBONES
# from ..utils import KerasMockLayer

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4


class BNReLULayer(tf.keras.layers.Layer):

    #def __call__(self, inputs, training=False, *args, **kwargs):


class GNReLULayer(tf.keras.layers.Layer):

    #def __call__(self, inputs, training=False, *args, **kwargs):
        

class FixedPaddingLayer(tf.keras.layers.Layer):

    #def __call__(self, inputs, *args, **kwargs):


class Conv2dFixedPadding(tf.keras.layers.Layer):

    #def __call__(self, inputs, *args, **kwargs):


class ResidualBlock(tf.keras.layers.Layer):

    #def __call__(self, inputs, training=False):


class BottleneckBlock(tf.keras.layers.Layer):

    #def __call__(self, inputs, training=False):


class BlockGroup(tf.keras.layers.Layer):

    #def __call__(self, inputs, training=False):

@BACKBONES.register("resnet18")
@BACKBONES.register("resnet34")
@BACKBONES.register("resnet50")
@BACKBONES.register("resnet101")
@BACKBONES.register("resnet152")
@BACKBONES.register("resnet200")
class Resnet_Model(tf.keras.models.Model):
            
            

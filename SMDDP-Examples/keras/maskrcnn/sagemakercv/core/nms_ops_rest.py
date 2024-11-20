#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tensorflow implementation of non max suppression."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import tensorflow as tf

from sagemakercv.core import box_utils


NMS_TILE_SIZE = 512









    selected_boxes, _, output_size, _ = tf.while_loop(
            _loop_cond, _suppression_loop_body, [
                    boxes, iou_threshold,
                    tf.zeros([batch_size], tf.int32),
                    tf.constant(0)
            ])
    idx = num_boxes - tf.cast(
            tf.nn.top_k(
                    tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
                    tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
            tf.int32)
    idx = tf.minimum(idx, num_boxes - 1)
    idx = tf.reshape(
            idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
    boxes = tf.reshape(
            tf.gather(tf.reshape(boxes, [-1, 4]), idx),
            [batch_size, max_output_size, 4])
    boxes = boxes * tf.cast(
            tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
                    output_size, [-1, 1, 1]), boxes.dtype)
    scores = tf.reshape(
            tf.gather(tf.reshape(scores, [-1, 1]), idx),
            [batch_size, max_output_size])
    scores = scores * tf.cast(
            tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
                    output_size, [-1, 1]), scores.dtype)
    return scores, boxes
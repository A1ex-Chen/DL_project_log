#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow as tf




class TfExampleDecoder(object):
    """Tensorflow Example proto decoder."""






        height = parsed_tensors['image/height']
        width = parsed_tensors['image/width']
        masks = parsed_tensors['image/object/mask']
        return tf.cond(
            tf.greater(tf.size(masks), 0),
            lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
            lambda: tf.zeros([0, height, width], dtype=tf.float32)
        )

    def decode(self, serialized_example):
        """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - image: a uint8 tensor of shape [None, None, 3].
        - source_id: a string scalar tensor.
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
        parsed_tensors = tf.io.parse_single_example(
            serialized_example, self._keys_to_features)
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value='')
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0)

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        is_crowd = tf.cast(parsed_tensors['image/object/is_crowd'], dtype=tf.bool)

        if self._include_mask:
            masks = self._decode_masks(parsed_tensors)

        if self._regenerate_source_id:
            source_id = _get_source_id_from_encoded_image(parsed_tensors)

        else:
            source_id = tf.cond(
                tf.greater(tf.strings.length(parsed_tensors['image/source_id']), 0),
                lambda: parsed_tensors['image/source_id'],
                lambda: _get_source_id_from_encoded_image(parsed_tensors)
            )

        decoded_tensors = {
            'image': image,
            # 'source_id': parsed_tensors['image/source_id'],
            'source_id': source_id,
            'height': parsed_tensors['image/height'],
            'width': parsed_tensors['image/width'],
            'groundtruth_classes': parsed_tensors['image/object/class/label'],
            'groundtruth_is_crowd': is_crowd,
            'groundtruth_area': parsed_tensors['image/object/area'],
            'groundtruth_boxes': boxes,
        }

        if self._include_mask:
            decoded_tensors.update({
                'groundtruth_instance_masks': masks,
                'groundtruth_instance_masks_png': parsed_tensors['image/object/mask'],
            })

        return decoded_tensors
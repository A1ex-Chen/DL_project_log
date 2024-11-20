#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Convert raw COCO dataset to TFRecord for object_detection.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import hashlib
import io
import json
import multiprocessing
import os
from absl import app
from absl import flags
import numpy as np
import PIL.Image

from pycocotools import mask
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

import tensorflow as tf

flags.DEFINE_boolean('include_masks', False,
                     'Whether to include instance segmentations masks '
                     '(PNG encoded) in the result. default: False.')
flags.DEFINE_string('train_image_dir', '', 'Training image directory.')
flags.DEFINE_string('val_image_dir', '', 'Validation image directory.')
flags.DEFINE_string('test_image_dir', '', 'Test image directory.')
flags.DEFINE_string('train_object_annotations_file', '', '')
flags.DEFINE_string('val_object_annotations_file', '', '')
flags.DEFINE_string('train_caption_annotations_file', '', '')
flags.DEFINE_string('val_caption_annotations_file', '', '')
flags.DEFINE_string('testdev_annotations_file', '',
                    'Test-dev annotations JSON file.')
flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)














if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  app.run(main)
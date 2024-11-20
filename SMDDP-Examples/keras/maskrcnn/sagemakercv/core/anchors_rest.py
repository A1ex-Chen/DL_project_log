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
"""Mask-RCNN anchor definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from sagemakercv.core import argmax_matcher
from sagemakercv.core import balanced_positive_negative_sampler
from sagemakercv.core import box_list
from sagemakercv.core import faster_rcnn_box_coder
from sagemakercv.core import region_similarity_calculator
from sagemakercv.core import target_assigner





class AnchorGenerator(object):
  """Mask-RCNN Anchors class."""







class AnchorLabeler(object):
  """Labeler for multiscale anchor boxes."""




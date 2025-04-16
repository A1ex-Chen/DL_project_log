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

"""Base minibatch sampler module.

The job of the minibatch_sampler is to subsample a minibatch based on some
criterion.

The main function call is:
    subsample(indicator, batch_size, **params).
Indicator is a 1d boolean tensor where True denotes which examples can be
sampled. It returns a boolean indicator where True denotes an example has been
sampled..

Subclasses should implement the Subsample function and can make use of the
@staticmethod SubsampleIndicator.

This is originally implemented in TensorFlow Object Detection API.
"""

from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from sagemakercv.core import ops


class MinibatchSampler(object):
  """Abstract base class for subsampling minibatches."""
  __metaclass__ = ABCMeta


  @abstractmethod

  @staticmethod

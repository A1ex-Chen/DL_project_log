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

"""Class to subsample minibatches by balancing positives and negatives.

Subsamples minibatches based on a pre-specified positive fraction in range
[0,1]. The class presumes there are many more negatives than positive examples:
if the desired batch_size cannot be achieved with the pre-specified positive
fraction, it fills the rest with negative examples. If this is not sufficient
for obtaining the desired batch_size, it returns fewer examples.

The main function to call is Subsample(self, indicator, labels). For convenience
one can also call SubsampleWeights(self, weights, labels) which is defined in
the minibatch_sampler base class.

When is_static is True, it implements a method that guarantees static shapes.
It also ensures the length of output of the subsample is always batch_size, even
when number of examples set to True in indicator is less than batch_size.

This is originally implemented in TensorFlow Object Detection API.
"""

import tensorflow as tf

from sagemakercv.core import minibatch_sampler
from sagemakercv.core import ops


class BalancedPositiveNegativeSampler(minibatch_sampler.MinibatchSampler):
  """Subsamples minibatches to a desired balance of positives and negatives."""






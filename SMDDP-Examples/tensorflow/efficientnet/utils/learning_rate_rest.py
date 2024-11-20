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
"""Learning rate utilities for vision tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, List, Mapping

import tensorflow as tf

BASE_LEARNING_RATE = 0.1

__all__ = [ 'WarmupDecaySchedule', 'PiecewiseConstantDecayWithWarmup' ]

@tf.keras.utils.register_keras_serializable(package='Custom')
class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A wrapper for LearningRateSchedule that includes warmup steps."""







# TODO(b/149030439) - refactor this with
# tf.keras.optimizers.schedules.PiecewiseConstantDecay + WarmupDecaySchedule.
class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup schedule."""



    return tf.cond(step < self._warmup_steps, warmup_lr, piecewise_lr)

  def get_config(self) -> Mapping[str, Any]:
    return {
        "rescaled_lr": self._rescaled_lr,
        "step_boundaries": self._step_boundaries,
        "lr_values": self._lr_values,
        "warmup_steps": self._warmup_steps,
    }
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os
import pytest

from transformers import is_torch_available

if is_torch_available():
    import torch

    from transformers import (AdamW,
                              get_constant_schedule,
                              get_constant_schedule_with_warmup,
                              get_cosine_schedule_with_warmup,
                              get_cosine_with_hard_restarts_schedule_with_warmup,
                              get_linear_schedule_with_warmup)
else:
    pytestmark = pytest.mark.skip("Require Torch")

from .tokenization_tests_commons import TemporaryDirectory




class OptimizationTest(unittest.TestCase):




class ScheduleInitTest(unittest.TestCase):
    m = torch.nn.Linear(50, 50) if is_torch_available() else None
    optimizer = AdamW(m.parameters(), lr=10.) if is_torch_available() else None
    num_steps = 10








if __name__ == "__main__":
    unittest.main()
# Copyright 2024 ParaDiGMS authors and The HuggingFace Team. All rights reserved.
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

import torch

from diffusers import DDIMParallelScheduler

from .test_schedulers import SchedulerCommonTest


class DDIMParallelSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDIMParallelScheduler,)
    forward_default_kwargs = (("eta", 0.0), ("num_inference_steps", 50))





















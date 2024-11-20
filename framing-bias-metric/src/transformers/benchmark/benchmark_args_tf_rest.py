# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import Tuple

from ..file_utils import cached_property, is_tf_available, tf_required
from ..utils import logging
from .benchmark_args_utils import BenchmarkArguments


if is_tf_available():
    import tensorflow as tf


logger = logging.get_logger(__name__)


@dataclass
class TensorFlowBenchmarkArguments(BenchmarkArguments):

    deprecated_args = [
        "no_inference",
        "no_cuda",
        "no_tpu",
        "no_speed",
        "no_memory",
        "no_env_print",
        "no_multi_process",
    ]


    tpu_name: str = field(
        default=None,
        metadata={"help": "Name of TPU"},
    )
    device_idx: int = field(
        default=0,
        metadata={"help": "CPU / GPU device index. Defaults to 0."},
    )
    eager_mode: bool = field(default=False, metadata={"help": "Benchmark models in eager model."})
    use_xla: bool = field(
        default=False,
        metadata={
            "help": "Benchmark models using XLA JIT compilation. Note that `eager_model` has to be set to `False`."
        },
    )

    @cached_property
    @tf_required

    @cached_property
    @tf_required

    @property
    @tf_required

    @property
    @tf_required

    @property
    @tf_required

    @property
    @tf_required

    @property
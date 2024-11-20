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

import abc
import importlib
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

LOGGER = logging.getLogger(__name__)
DATALOADER_FN_NAME = "get_dataloader_fn"
GET_MODEL_FN_NAME = "get_model"
GET_SERVING_INPUT_RECEIVER_FN = "get_serving_input_receiver_fn"
GET_ARGPARSER_FN_NAME = "update_argparser"


class TensorSpec(NamedTuple):
    name: str
    dtype: str
    shape: Tuple


class Parameter(Enum):
    def __lt__(self, other: "Parameter") -> bool:
        return self.value < other.value


class Accelerator(Parameter):
    AMP = "amp"
    CUDA = "cuda"
    TRT = "trt"


class Precision(Parameter):
    FP16 = "fp16"
    FP32 = "fp32"
    TF32 = "tf32"  # Deprecated


class Format(Parameter):
    TF_GRAPHDEF = "tf-graphdef"
    TF_SAVEDMODEL = "tf-savedmodel"
    TF_TRT = "tf-trt"
    TF_ESTIMATOR = "tf-estimator"
    TF_KERAS = "tf-keras"
    ONNX = "onnx"
    TRT = "trt"
    TS_SCRIPT = "ts-script"
    TS_TRACE = "ts-trace"
    PYT = "pyt"


class Model(NamedTuple):
    handle: object
    precision: Optional[Precision]
    inputs: Dict[str, TensorSpec]
    outputs: Dict[str, TensorSpec]




class Accelerator(Parameter):
    AMP = "amp"
    CUDA = "cuda"
    TRT = "trt"


class Precision(Parameter):
    FP16 = "fp16"
    FP32 = "fp32"
    TF32 = "tf32"  # Deprecated


class Format(Parameter):
    TF_GRAPHDEF = "tf-graphdef"
    TF_SAVEDMODEL = "tf-savedmodel"
    TF_TRT = "tf-trt"
    TF_ESTIMATOR = "tf-estimator"
    TF_KERAS = "tf-keras"
    ONNX = "onnx"
    TRT = "trt"
    TS_SCRIPT = "ts-script"
    TS_TRACE = "ts-trace"
    PYT = "pyt"


class Model(NamedTuple):
    handle: object
    precision: Optional[Precision]
    inputs: Dict[str, TensorSpec]
    outputs: Dict[str, TensorSpec]


def load_from_file(file_path, label, target):
    spec = importlib.util.spec_from_file_location(name=label, location=file_path)
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)  # pytype: disable=attribute-error
    return getattr(my_module, target, None)


class BaseLoader(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod


class BaseSaver(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod


class BaseRunner(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod


class BaseRunnerSession(abc.ABC):

    @abc.abstractmethod

    @abc.abstractmethod

    @abc.abstractmethod




class BaseConverter(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod

    @staticmethod


class BaseMetricsCalculator(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod


class ShapeSpec(NamedTuple):
    min: Tuple
    opt: Tuple
    max: Tuple
#!/usr/bin/env python3

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
r"""
`convert_model.py` script allows to convert between model formats with additional model optimizations
for faster inference.
It converts model from results of get_model function.

Currently supported input and output formats are:

  - inputs
    - `tf-estimator` - `get_model` function returning Tensorflow Estimator
    - `tf-keras` - `get_model` function returning Tensorflow Keras Model
    - `tf-savedmodel` - Tensorflow SavedModel binary
    - `pyt` - `get_model` function returning PyTorch Module
  - output
    - `tf-savedmodel` - Tensorflow saved model
    - `tf-trt` - TF-TRT saved model
    - `ts-trace` - PyTorch traced ScriptModule
    - `ts-script` - PyTorch scripted ScriptModule
    - `onnx` - ONNX
    - `trt` - TensorRT plan file

For tf-keras input you can use:
  - --large-model flag - helps loading model which exceeds maximum protobuf size of 2GB
  - --tf-allow-growth flag - control limiting GPU memory growth feature
    (https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth). By default it is disabled.
"""

import argparse
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator
from .deployment_toolkit.core import (
    DATALOADER_FN_NAME,
    BaseConverter,
    BaseLoader,
    BaseSaver,
    Format,
    Precision,
    load_from_file,
)
from .deployment_toolkit.extensions import converters, loaders, savers

LOGGER = logging.getLogger("convert_model")

INPUT_MODEL_TYPES = [Format.TF_ESTIMATOR, Format.TF_KERAS, Format.TF_SAVEDMODEL, Format.PYT]
OUTPUT_MODEL_TYPES = [Format.TF_SAVEDMODEL, Format.TF_TRT, Format.ONNX, Format.TRT, Format.TS_TRACE, Format.TS_SCRIPT]






if __name__ == "__main__":
    main()
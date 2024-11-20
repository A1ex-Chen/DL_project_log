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
To infer the model on framework runtime, you can use `run_inference_on_fw.py` script.
It infers data obtained from pointed data loader locally and saves received data into npz files.
Those files are stored in directory pointed by `--output-dir` argument.

Example call:

```shell script
python ./triton/run_inference_on_fw.py \
    --input-path /models/exported/model.onnx \
    --input-type onnx \
    --dataloader triton/dataloader.py \
    --data-dir /data/imagenet \
    --batch-size 32 \
    --output-dir /results/dump_local \
    --dump-labels
```
"""

import argparse
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "0"

from tqdm import tqdm

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator
from .deployment_toolkit.core import DATALOADER_FN_NAME, BaseLoader, BaseRunner, Format, load_from_file
from .deployment_toolkit.dump import NpzWriter
from .deployment_toolkit.extensions import loaders, runners

LOGGER = logging.getLogger("run_inference_on_fw")








if __name__ == "__main__":
    main()
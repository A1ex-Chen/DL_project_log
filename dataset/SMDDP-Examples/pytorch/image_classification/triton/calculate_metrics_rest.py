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
Using `calculate_metrics.py` script, you can obtain model accuracy/error metrics using defined `MetricsCalculator` class.

Data provided to `MetricsCalculator` are obtained from npz dump files
stored in directory pointed by `--dump-dir` argument.
Above files are prepared by `run_inference_on_fw.py` and `run_inference_on_triton.py` scripts.

Output data is stored in csv file pointed by `--csv` argument.

Example call:

```shell script
python ./triton/calculate_metrics.py \
    --dump-dir /results/dump_triton \
    --csv /results/accuracy_results.csv \
    --metrics metrics.py \
    --metric-class-param1 value
```
"""

import argparse
import csv
import logging
import string
from pathlib import Path

import numpy as np

# method from PEP-366 to support relative import in executed modules

if __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator
from .deployment_toolkit.core import BaseMetricsCalculator, load_from_file
from .deployment_toolkit.dump import pad_except_batch_axis

LOGGER = logging.getLogger("calculate_metrics")
TOTAL_COLUMN_NAME = "_total_"






if __name__ == "__main__":
    main()
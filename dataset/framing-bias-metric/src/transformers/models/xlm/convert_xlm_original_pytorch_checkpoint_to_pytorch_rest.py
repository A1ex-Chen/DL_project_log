# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert OpenAI GPT checkpoint."""


import argparse
import json

import numpy
import torch

from transformers import CONFIG_NAME, WEIGHTS_NAME
from transformers.models.xlm.tokenization_xlm import VOCAB_FILES_NAMES
from transformers.utils import logging


logging.set_verbosity_info()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--xlm_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_xlm_checkpoint_to_pytorch(args.xlm_checkpoint_path, args.pytorch_dump_folder_path)
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert BART checkpoint."""


import argparse
import os
from pathlib import Path

import fairseq
import torch
from packaging import version

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from transformers.models.bart.modeling_bart import _make_linear_from_emb
from transformers.utils import logging


FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
extra_arch = {"bart.large": BartModel, "bart.large.mnli": BartForSequenceClassification}
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]








@torch.no_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config", default=None, type=str, help="Which huggingface architecture to use: bart-large-xsum"
    )
    args = parser.parse_args()
    convert_bart_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config)
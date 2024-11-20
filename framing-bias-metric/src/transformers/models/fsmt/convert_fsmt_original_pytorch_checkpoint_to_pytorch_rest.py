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

# Note: if you intend to run this script make sure you look under scripts/fsmt/
# to locate the appropriate script to do the work correctly. There is a set of scripts to:
# - download and prepare data and run the conversion script
# - perform eval to get the best hparam into the config
# - generate model_cards - useful if you have multiple models from the same paper

import argparse
import json
import os
import re
from collections import OrderedDict
from os.path import basename, dirname

import fairseq
import torch
from fairseq import hub_utils
from fairseq.data.dictionary import Dictionary

from transformers import WEIGHTS_NAME, logging
from transformers.models.fsmt import VOCAB_FILES_NAMES, FSMTConfig, FSMTForConditionalGeneration
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE


logging.set_verbosity_warning()

json_indent = 2

# based on the results of a search on a range of `num_beams`, `length_penalty` and `early_stopping`
# values against wmt19 test data to obtain the best BLEU scores, we will use the following defaults:
#
# * `num_beams`: 5 (higher scores better, but requires more memory/is slower, can be adjusted by users)
# * `early_stopping`: `False` consistently scored better
# * `length_penalty` varied, so will assign the best one depending on the model
best_score_hparams = {
    # fairseq:
    "wmt19-ru-en": {"length_penalty": 1.1},
    "wmt19-en-ru": {"length_penalty": 1.15},
    "wmt19-en-de": {"length_penalty": 1.0},
    "wmt19-de-en": {"length_penalty": 1.1},
    # allenai:
    "wmt16-en-de-dist-12-1": {"length_penalty": 0.6},
    "wmt16-en-de-dist-6-1": {"length_penalty": 0.6},
    "wmt16-en-de-12-1": {"length_penalty": 0.8},
    "wmt19-de-en-6-6-base": {"length_penalty": 0.6},
    "wmt19-de-en-6-6-big": {"length_penalty": 0.6},
}

# this remaps the different models to their organization names
org_names = {}
for m in ["wmt19-ru-en", "wmt19-en-ru", "wmt19-en-de", "wmt19-de-en"]:
    org_names[m] = "facebook"
for m in [
    "wmt16-en-de-dist-12-1",
    "wmt16-en-de-dist-6-1",
    "wmt16-en-de-12-1",
    "wmt19-de-en-6-6-base",
    "wmt19-de-en-6-6-big",
]:
    org_names[m] = "allenai"






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fsmt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the official PyTorch checkpoint file which is expected to reside in the dump dir with dicts, bpecodes, etc.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_fsmt_checkpoint_to_pytorch(args.fsmt_checkpoint_path, args.pytorch_dump_folder_path)
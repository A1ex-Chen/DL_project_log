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
""" Convert slow tokenizers checkpoints in fast (serialization format of the `tokenizers` library) """

import argparse
import os

import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from transformers.utils import logging


logging.set_verbosity_info()

logger = logging.get_logger(__name__)


TOKENIZER_CLASSES = {name: getattr(transformers, name + "Fast") for name in SLOW_TO_FAST_CONVERTERS}




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to output generated fast tokenizer files."
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional tokenizer type selected in the list of {}. If not given, will download and convert all the checkpoints from AWS.".format(
            list(TOKENIZER_CLASSES.keys())
        ),
    )
    parser.add_argument(
        "--checkpoint_name",
        default=None,
        type=str,
        help="Optional checkpoint name. If not given, will download and convert the canonical checkpoints from AWS.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download checkpoints.",
    )
    args = parser.parse_args()

    convert_slow_checkpoint_to_fast(args.tokenizer_name, args.checkpoint_name, args.dump_path, args.force_download)
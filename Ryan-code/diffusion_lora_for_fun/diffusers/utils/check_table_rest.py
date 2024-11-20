# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

import argparse
import collections
import importlib.util
import os
import re


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_table.py
TRANSFORMERS_PATH = "src/diffusers"
PATH_TO_DOCS = "docs/source/en"
REPO_PATH = "."




# Add here suffixes that are used to identify models, separated by |
ALLOWED_MODEL_SUFFIXES = "Model|Encoder|Decoder|ForConditionalGeneration"
# Regexes that match TF/Flax/PT model names.
_re_tf_models = re.compile(r"TF(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
_re_flax_models = re.compile(r"Flax(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
# Will match any TF or Flax model too so need to be in an else branch afterthe two previous regexes.
_re_pt_models = re.compile(r"(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")


# This is to make sure the diffusers module imported is the one in the repo.
spec = importlib.util.spec_from_file_location(
    "diffusers",
    os.path.join(TRANSFORMERS_PATH, "__init__.py"),
    submodule_search_locations=[TRANSFORMERS_PATH],
)
diffusers_module = spec.loader.load_module()


# Thanks to https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_model_table(args.fix_and_overwrite)
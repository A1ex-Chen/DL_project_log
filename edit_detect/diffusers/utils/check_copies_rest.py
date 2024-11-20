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
import glob
import os
import re
import subprocess


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_copies.py
DIFFUSERS_PATH = "src/diffusers"
REPO_PATH = "."






_re_copy_warning = re.compile(r"^(\s*)#\s*Copied from\s+diffusers\.(\S+\.\S+)\s*($|\S.*$)")
_re_replace_pattern = re.compile(r"^\s*(\S+)->(\S+)(\s+.*|$)")
_re_fill_pattern = re.compile(r"<FILL\s+[^>]*>")












if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix_and_overwrite",
        action="store_true",
        help="Whether to fix inconsistencies.",
    )
    args = parser.parse_args()

    check_copies(args.fix_and_overwrite)
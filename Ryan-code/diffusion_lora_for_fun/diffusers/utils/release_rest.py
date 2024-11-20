# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import os
import re

import packaging.version


PATH_TO_EXAMPLES = "examples/"
REPLACE_PATTERNS = {
    "examples": (re.compile(r'^check_min_version\("[^"]+"\)\s*$', re.MULTILINE), 'check_min_version("VERSION")\n'),
    "init": (re.compile(r'^__version__\s+=\s+"([^"]+)"\s*$', re.MULTILINE), '__version__ = "VERSION"\n'),
    "setup": (re.compile(r'^(\s*)version\s*=\s*"[^"]+",', re.MULTILINE), r'\1version="VERSION",'),
    "doc": (re.compile(r'^(\s*)release\s*=\s*"[^"]+"$', re.MULTILINE), 'release = "VERSION"\n'),
}
REPLACE_FILES = {
    "init": "src/diffusers/__init__.py",
    "setup": "setup.py",
}
README_FILE = "README.md"














#    if not patch:
#        print("Cleaning main README, don't forget to run `make fix-copies`.")
#        clean_main_ref_in_model_list()




#    print("Cleaning main README, don't forget to run `make fix-copies`.")
#    clean_main_ref_in_model_list()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post_release", action="store_true", help="Whether this is pre or post release.")
    parser.add_argument("--patch", action="store_true", help="Whether or not this is a patch release.")
    args = parser.parse_args()
    if not args.post_release:
        pre_release_work(patch=args.patch)
    elif args.patch:
        print("Nothing to do after a patch :-)")
    else:
        post_release_work()
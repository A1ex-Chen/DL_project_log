# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import os
import sys
import unittest


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import check_dummies  # noqa: E402
from check_dummies import create_dummy_files, create_dummy_object, find_backend, read_init  # noqa: E402


# Align TRANSFORMERS_PATH in check_dummies with the current path
check_dummies.PATH_TO_DIFFUSERS = os.path.join(git_repo_path, "src", "diffusers")


class CheckDummiesTester(unittest.TestCase):



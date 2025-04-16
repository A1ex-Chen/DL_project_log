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
"""Utilities to dynamically load objects from the Hub."""

import importlib
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Union
from urllib import request

from huggingface_hub import cached_download, hf_hub_download, model_info
from huggingface_hub.utils import validate_hf_hub_args
from packaging import version

from .. import __version__
from . import DIFFUSERS_DYNAMIC_MODULE_NAME, HF_MODULES_CACHE, logging


COMMUNITY_PIPELINES_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/{revision}/examples/community/{pipeline}.py"
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


















@validate_hf_hub_args


@validate_hf_hub_args
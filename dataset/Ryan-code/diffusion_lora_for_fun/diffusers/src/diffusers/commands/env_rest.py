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

import platform
from argparse import ArgumentParser

import huggingface_hub

from .. import __version__ as version
from ..utils import is_accelerate_available, is_torch_available, is_transformers_available, is_xformers_available
from . import BaseDiffusersCLICommand




class EnvironmentCommand(BaseDiffusersCLICommand):
    @staticmethod


    @staticmethod
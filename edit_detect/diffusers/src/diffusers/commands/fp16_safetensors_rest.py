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

"""
Usage example:
    diffusers-cli fp16_safetensors --ckpt_id=openai/shap-e --fp16 --use_safetensors
"""

import glob
import json
import warnings
from argparse import ArgumentParser, Namespace
from importlib import import_module

import huggingface_hub
import torch
from huggingface_hub import hf_hub_download
from packaging import version

from ..utils import logging
from . import BaseDiffusersCLICommand




class FP16SafetensorsCommand(BaseDiffusersCLICommand):
    @staticmethod


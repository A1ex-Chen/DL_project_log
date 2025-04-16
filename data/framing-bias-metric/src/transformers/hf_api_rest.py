# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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


import io
import os
from os.path import expanduser
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import requests


ENDPOINT = "https://huggingface.co"


class RepoObj:
    """
    HuggingFace git-based system, data structure that represents a file belonging to the current user.
    """



class S3Obj:
    """
    HuggingFace S3-based system, data structure that represents a file belonging to the current user.
    """



class PresignedUrl:


class ModelSibling:
    """
    Data structure that represents a public file inside a model, accessible from huggingface.co
    """



class ModelInfo:
    """
    Info about a public model accessible from huggingface.co
    """



class HfApi:













class TqdmProgressFileReader:
    """
    Wrap an io.BufferedReader `f` (such as the output of `open(…, "rb")`) and override `f.read()` so as to display a
    tqdm progress bar.

    see github.com/huggingface/transformers/pull/2078#discussion_r354739608 for implementation details.
    """





class HfFolder:
    path_token = expanduser("~/.huggingface/token")

    @classmethod

    @classmethod

    @classmethod
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import importlib
import logging
import os
import re
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)


class ExtensionManager:



    @property

    @staticmethod


runners = ExtensionManager("runners")
loaders = ExtensionManager("loaders")
savers = ExtensionManager("savers")
converters = ExtensionManager("converters")
toolkit_root_dir = (Path(__file__).parent / "..").resolve()
ExtensionManager.scan_for_extensions([toolkit_root_dir])
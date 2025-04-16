"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import io
import json
import logging
import os
import pickle
import re
import shutil
import urllib
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml
from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr
from ..common.registry import registry
from torch.utils.model_zoo import tqdm
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    extract_archive,
)












# The following are adapted from torchvision and vissl
# torchvision: https://github.com/pytorch/vision
# vissl: https://github.com/facebookresearch/vissl/blob/main/vissl/utils/download.py




















# TODO (prigoyal): convert this into RAII-style API














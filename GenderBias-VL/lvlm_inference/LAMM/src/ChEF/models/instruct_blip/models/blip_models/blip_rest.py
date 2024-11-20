"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
from packaging import version

import torch
from ...common.dist_utils import download_cached_file
from ...common.utils import is_url
from ..base_model import BaseModel
from ..vit import interpolate_pos_embed
from transformers import BertTokenizer
import transformers

class BlipBase(BaseModel):
        
    @classmethod

import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path

import torch

from .model import CLAP, convert_weights_to_fp16
from .openai import load_openai_model
from .pretrained import get_pretrained_url, download_pretrained
from .transform import image_transform

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs






_rescan_model_configs()  # initial populate of model config registry










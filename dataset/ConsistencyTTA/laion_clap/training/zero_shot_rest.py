# NOTE: This script is currently not supported for CLAP.
import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

from clap_module import tokenize
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template








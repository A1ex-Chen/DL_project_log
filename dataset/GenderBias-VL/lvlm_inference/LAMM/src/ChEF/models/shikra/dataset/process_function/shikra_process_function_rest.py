import sys
import copy
import warnings
import logging
from typing import Dict, Any, List

import PIL.Image
import torch
from PIL import Image
from transformers import LlamaTokenizer

from ..root import (
    FUNCTIONS,
    IMAGE_PLACEHOLDER,
    BaseImageProcessFunc,
    BaseConvProcessFunc,
    BaseTextProcessFunc,
)
from ...conversation import SeparatorStyle, Conversation

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = IMAGE_PLACEHOLDER
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@FUNCTIONS.register_module()
class ShikraConvProcess(BaseConvProcessFunc):


@FUNCTIONS.register_module()
class ShikraTextProcess(BaseTextProcessFunc):


    # noinspection PyMethodMayBeStatic

    # noinspection PyMethodMayBeStatic


@FUNCTIONS.register_module()
class ShikraImageProcessor(BaseImageProcessFunc):
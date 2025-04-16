import warnings
from functools import partial
from typing import Dict, Any, Callable, List, Optional, Tuple, Type

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainingArguments

from .root import IMAGE_PLACEHOLDER, BOXES_PLACEHOLDER
from ..conversation import Conversation, get_conv_template
from ..utils import post_process_generate_ids


class SingleImageConvDatasetMixin:




    # noinspection PyMethodMayBeStatic


    # noinspection PyMethodMayBeStatic
        # not check box placeholder num this will be checked in format process








class SingleImageConvDataset(SingleImageConvDatasetMixin, Dataset):
    _repr_indent = 4







__all__ = ['SingleImageConvDatasetMixin', 'SingleImageConvDataset']
import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPVisionModelWithProjection, PreTrainedModel

from ...utils import logging


logger = logging.get_logger(__name__)


class IFSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]


    @torch.no_grad()
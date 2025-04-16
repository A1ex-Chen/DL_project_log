""" CokeBert model configuration """

from collections import OrderedDict
from typing import Mapping

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class CokeBertConfig(PretrainedConfig):
    model_type = "coke_bert"



class CokeBertOnnxConfig(OnnxConfig):
    @property
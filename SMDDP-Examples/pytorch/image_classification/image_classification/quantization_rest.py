from tqdm import tqdm
import torch
import contextlib
import time
import logging

from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from . import logger as log
from .utils import calc_ips
import dllogger

initialize = quant_modules.initialize
deactivate = quant_modules.deactivate

IPS_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}














@contextlib.contextmanager
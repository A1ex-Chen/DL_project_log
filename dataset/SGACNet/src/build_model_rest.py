# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings

import torch
from torch import nn

from src.models.model import SGACNet
from src.models.model_one_modality import SGACNetOneModality
from src.models.resnet import ResNet


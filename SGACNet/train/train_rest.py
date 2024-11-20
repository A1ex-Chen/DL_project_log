# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from datetime import datetime
import json
import pickle
import os
import sys
import time
import warnings

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src import utils
from src.prepare_data import prepare_data
from src.utils import save_ckpt, save_ckpt_every_epoch
from src.utils import load_ckpt
from src.utils import print_log
from src.utils import netParams
from src.logger import CSVLogger
from src.confusion_matrix import ConfusionMatrixTensorflow
GLOBAL_SEED = 1234











if __name__ == '__main__':
    train_main()
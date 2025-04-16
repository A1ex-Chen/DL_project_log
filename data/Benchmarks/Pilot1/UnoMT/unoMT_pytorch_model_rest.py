"""
    File Name:          UnoPytorch/uno_pytorch.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   This is a version of the original file
                        modified to fit CANDLE framework.
                        Date: 3/12/19.

"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.functions.cl_clf_func import train_cl_clf, valid_cl_clf
from networks.functions.drug_qed_func import train_drug_qed, valid_drug_qed
from networks.functions.drug_target_func import train_drug_target, valid_drug_target
from networks.functions.resp_func import train_resp, valid_resp
from networks.initialization.encoder_init import get_drug_encoder, get_gene_encoder
from networks.structures.classification_net import ClfNet
from networks.structures.regression_net import RgsNet
from networks.structures.response_net import RespNet
from torch.optim.lr_scheduler import LambdaLR
from utils.data_processing.label_encoding import get_label_dict
from utils.datasets.cl_class_dataset import CLClassDataset
from utils.datasets.drug_qed_dataset import DrugQEDDataset
from utils.datasets.drug_resp_dataset import DrugRespDataset
from utils.datasets.drug_target_dataset import DrugTargetDataset
from utils.miscellaneous.optimizer import get_optimizer

# Number of workers for dataloader. Too many workers might lead to process
# hanging for PyTorch version 4.1. Set this number between 0 and 4.
NUM_WORKER = 4
DATA_ROOT = "../../Data/Pilot1/"


class UnoMTModel(object):










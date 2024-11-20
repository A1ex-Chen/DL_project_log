"""
    File Name:          UnoPytorch/encoder_init.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/16/18
    Python Version:     3.6.6
    File Description:

"""
import copy
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from networks.structures.encoder_net import EncNet
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from utils.data_processing.cell_line_dataframes import get_rna_seq_df
from utils.data_processing.drug_dataframes import get_drug_feature_df
from utils.datasets.basic_dataset import DataFrameDataset
from utils.miscellaneous.optimizer import get_optimizer
from utils.miscellaneous.random_seeding import seed_random_state

logger = logging.getLogger(__name__)








if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test code for autoencoder with RNA sequence and drug features
    ae_training_kwarg = {
        "ae_loss_func": "mse",
        "ae_opt": "sgd",
        "ae_lr": 0.2,
        "lr_decay_factor": 1.0,
        "max_num_epochs": 1000,
        "early_stop_patience": 50,
    }

    gene_encoder = get_gene_encoder(
        model_folder="../../models/",
        data_root="../../data/",
        rnaseq_feature_usage="source_scale",
        rnaseq_scaling="std",
        autoencoder_init=True,
        layer_dim=1024,
        num_layers=2,
        latent_dim=512,
        training_kwarg=ae_training_kwarg,
        device=torch.device("cuda"),
        verbose=True,
        rand_state=0,
    )

    drug_encoder = get_drug_encoder(
        model_folder="../../models/",
        data_root="../../data/",
        drug_feature_usage="both",
        dscptr_scaling="std",
        dscptr_nan_threshold=0.0,
        autoencoder_init=True,
        layer_dim=4096,
        num_layers=2,
        latent_dim=2048,
        training_kwarg=ae_training_kwarg,
        device=torch.device("cuda"),
        verbose=True,
        rand_state=0,
    )
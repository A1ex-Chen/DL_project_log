"""
    File Name:          UnoPytorch/drug_target_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:

"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from utils.data_processing.drug_dataframes import (
    get_drug_feature_df,
    get_drug_target_df,
)

logger = logging.getLogger(__name__)


class DrugTargetDataset(data.Dataset):





if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    # Test DrugRespDataset class
    dataloader = torch.utils.data.DataLoader(
        DrugTargetDataset(data_root="../../data/", training=True),
        batch_size=512,
        shuffle=False,
    )

    tmp = dataloader.dataset[0]
    print(tmp)
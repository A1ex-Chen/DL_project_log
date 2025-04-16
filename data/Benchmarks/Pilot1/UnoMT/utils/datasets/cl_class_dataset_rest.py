"""
    File Name:          UnoPytorch/cl_class_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:
        This file implements the dataset for cell line classification.
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from utils.data_processing.cell_line_dataframes import get_cl_meta_df, get_rna_seq_df
from utils.data_processing.label_encoding import encode_int_to_onehot, get_label_dict

logger = logging.getLogger(__name__)


class CLClassDataset(data.Dataset):
    """Dataset class for cell line classification

    This class implements a PyTorch Dataset class made for cell line
    classification. Using enumerate() or any other methods that utilize
    __getitem__() to access the data.

    Each data item is made of a tuple of
        (RNA_sequence, conditions, site, type, category)
    where conditions is a list of [data_source, cell_description].

    Note that all categorical labels are numeric, and the encoding
    dictionary can be found in the processed folder.

    Attributes:
        training (bool): indicator of training/validation dataset
        cells (list): list of all the cells in the dataset
        num_cells (int): number of cell lines in the dataset
        rnaseq_dim (int): dimensionality of RNA sequence
    """






# Test segment for cell line classification dataset
if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    # Test DrugRespDataset class
    dataloader = torch.utils.data.DataLoader(
        CLClassDataset(data_root="../../data/", training=False),
        batch_size=512,
        shuffle=False,
    )

    tmp = dataloader.dataset[0]
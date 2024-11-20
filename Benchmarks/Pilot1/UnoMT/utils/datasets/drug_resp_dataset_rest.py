"""
    File Name:          UnoPytorch/drug_resp_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:
        This file implements the dataset for drug response.
"""

import logging

import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from utils.data_processing.cell_line_dataframes import get_cl_meta_df, get_rna_seq_df
from utils.data_processing.drug_dataframes import get_drug_feature_df
from utils.data_processing.label_encoding import get_label_dict
from utils.data_processing.response_dataframes import (
    get_drug_anlys_df,
    get_drug_resp_df,
)

logger = logging.getLogger(__name__)


class DrugRespDataset(data.Dataset):
    """Dataset class for drug response learning.

    This class implements a PyTorch Dataset class made for drug response
    learning. Using enumerate() or any other methods that utilize
    __getitem__() to access the data.

    Each data item is made of a tuple of (feature, target), where feature is
    a list including drug and cell line information along with the log
    concentration, and target is the growth.

    Note that all items in feature and the target are in python float type.

    Attributes:
        training (bool): indicator of training/validation dataset.
        drugs (list): list of all the drugs in the dataset.
        cells (list): list of all the cells in the dataset.
        data_source (str): source of the data being used.
        num_records (int): number of drug response records.
        drug_feature_dim (int): dimensionality of drug feature.
        rnaseq_dim (int): dimensionality of RNA sequence.
    """







# Test segment for drug response dataset
if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    for src in ["NCI60", "CTRP", "GDSC", "CCLE", "gCSI"]:

        kwarg = {
            "data_root": "../../data/",
            "summary": False,
            "rand_state": 0,
        }

        trn_set = DrugRespDataset(data_src=src, training=True, **kwarg)

        val_set = DrugRespDataset(data_src=src, training=False, **kwarg)

        print(
            "There are %i drugs and %i cell lines in %s."
            % (
                (len(trn_set.drugs) + len(val_set.drugs)),
                (len(trn_set.cells) + len(val_set.cells)),
                src,
            )
        )
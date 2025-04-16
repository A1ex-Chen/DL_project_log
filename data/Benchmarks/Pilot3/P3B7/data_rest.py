import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class P3B3(Dataset):
    """P3B3 Synthetic Dataset.

    Args:
        root: str
            Root directory of dataset where CANDLE loads P3B3 data.

        partition: str
            dataset partition to be loaded.
            Must be either 'train' or 'test'.
    """

    training_data_file = "train_X.npy"
    training_label_file = "train_Y.npy"
    test_data_file = "test_X.npy"
    test_label_file = "test_Y.npy"








class Vocabulary:




class Tokenizer:




class Egress(Dataset):
    r"""Static split from HJ's data handler

    Targets have six classes, with the following number of classes:

    site: 70,
    subsite: 325,
    laterality: 7,
    histology: 575,
    behaviour: 4,
    grade: 9

    Args:
        root: path to store the data
        split: Split to load. Either 'train' or 'valid'
    """

    store = Path("/gpfs/alpine/proj-shared/med107/NCI_Data/yngtodd/dat.pickle")















import os

import numpy as np
from torch.utils.data import Dataset


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






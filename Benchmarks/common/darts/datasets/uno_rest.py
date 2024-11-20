import os

import numpy as np
import pandas as pd
import torch
from darts.api import InMemoryDataset
from darts.datasets.utils import download_url, makedir_exist_ok


class Uno(InMemoryDataset):
    """Uno Dataset

    Parameters
    ----------
    root str :
        Root directory of dataset where ``processed/training.npy``
        ``processed/validation.npy and ``processed/test.npy`` exist.

    partition : str
        dataset partition to be loaded.
        Either 'train', 'validation', or 'test'.

    download : bool, optional
        If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    """

    urls = [
        "http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5",
    ]

    training_data_file = "train_data.pt"
    training_label_file = "train_labels.pt"
    test_data_file = "test_data.pt"
    test_label_file = "test_labels.pt"







    @property

    @property


    @staticmethod


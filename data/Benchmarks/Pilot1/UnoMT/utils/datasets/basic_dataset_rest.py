"""
    File Name:          UnoPytorch/basic_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/16/18
    Python Version:     3.6.6
    File Description:

"""

import numpy as np
import pandas as pd
import torch.utils.data as data


class DataFrameDataset(data.Dataset):
    """This class implements a basic PyTorch dataset from given dataframe.

    Note that this class does not take care of any form of data processing.
    It merely stores data from given dataframe in the form of np.array in
    certain data type.

    Also note that the given dataframe should be purely numeric.
    """



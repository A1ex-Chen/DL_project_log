"""
    File Name:          UnoPytorch/dataframe_scaling.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:

"""

import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)
SCALING_METHODS = ["none", "std", "minmax"]


#! /usr/bin/env python

import argparse
import os
import random

import numpy as np
import pandas as pd
from skwrapper import classify, regress, split_data, train

# MODELS = ['LightGBM', 'XGBoost', 'RandomForest']
MODELS = ["LightGBM"]
CV = 3
THREADS = 4
OUT_DIR = "p1save"
BINS = 0
CUTOFFS = None
FEATURE_SUBSAMPLE = 0
SEED = 2018








if __name__ == "__main__":
    main()
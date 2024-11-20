from __future__ import print_function

import collections
import os

import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer as Imputer
except ImportError:
    from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

global_cache = {}

SEED = 2017
P1B3_URL = "http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/"
DATA_URL = "http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/"












































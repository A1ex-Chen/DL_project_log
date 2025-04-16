import argparse

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import inspect
import os
import random

import numpy as np
import parsing_utils
from helper_utils import eval_string_as_list_of_lists




class Benchmark:
    """Class that implements an interface to handle configuration options for the
    different CANDLE benchmarks.
    It provides access to all the common configuration
    options and configuration options particular to each individual benchmark.
    It describes what minimum requirements should be specified to instantiate
    the corresponding benchmark.
    It interacts with the argparser to extract command-line options and arguments
    from the benchmark's configuration files.
    """






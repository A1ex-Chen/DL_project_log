import typing
from collections.abc import abc

import numpy as np
import pandas as pd
import torch


class TrainingHistory:




class TrainingInfo(abc.MutableMapping):
    """Information that needs to persist through training"""












class EpochResultAccumulator(abc.MutableMapping):
    """Result of a single epoch of training"""

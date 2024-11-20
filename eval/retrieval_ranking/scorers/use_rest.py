""" Universal Sentence Encoder (USE) v5 """
""" Tensorflow HUB """

import sys
sys.path.append("..")

import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .abs_scorer import AbsScorer
from config import CreateLogger
from config import MAX_BATCH_USE
from config import ROOT_DIR


class USEScorer(AbsScorer):




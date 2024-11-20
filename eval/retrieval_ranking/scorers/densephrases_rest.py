""" DensePhrases model for Phrase search   """
""" https://arxiv.org/abs/2012.12624 """

import sys
sys.path.append("..")

import spacy
import torch

from .abs_scorer import AbsScorer
from densephrases import DensePhrases
from config import CreateLogger
from config import MAX_BATCH_DENSEPHRASES


class DensePhrasesScorer(AbsScorer):





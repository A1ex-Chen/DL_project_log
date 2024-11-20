""" SimCSE model for Phrase search   """
""" https://arxiv.org/abs/2104.08821 """

import sys
sys.path.append("..")

import spacy
import torch
from simcse import SimCSE

from .abs_scorer import AbsScorer
from config import CreateLogger
from config import MAX_BATCH_SimCSE


class SimCSEScorer(AbsScorer):



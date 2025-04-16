""" Bert model for Phrase search """

import sys
sys.path.append("..")

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

from .abs_scorer import AbsScorer
from config import CreateLogger
from config import MAX_BATCH_BERT


class BertScorer(AbsScorer):





""" span bert (tacl-20) model for Phrase search """

import sys
sys.path.append("..")

import spacy
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

from .abs_scorer import AbsScorer
from config import CreateLogger
from config import MAX_BATCH_SPANBERT


class SpanBertScorer(AbsScorer):



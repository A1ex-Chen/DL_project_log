""" sentence bert (emnlp-19) model for Phrase search """

import sys
sys.path.append("..")

import spacy
import torch
from sentence_transformers import SentenceTransformer

from .abs_scorer import AbsScorer
from config import CreateLogger
from config import MAX_BATCH_SENTENCEBERT


class SentenceBertScorer(AbsScorer):






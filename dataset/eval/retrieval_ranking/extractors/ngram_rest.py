import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import spacy
from nltk.util import ngrams
from config import CreateLogger
from commons import AbsExtractor


class NGramExtractor(AbsExtractor):
    





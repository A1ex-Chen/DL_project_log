""" """
from commons import AbsSearch
from config import CreateLogger
from config import REMOVE_CANDIDATE_STOPWORDS

from scorers import BertScorer, SentenceBertScorer, SpanBertScorer, USEScorer, SimCSEScorer, DensePhrasesScorer
from extractors import NGramExtractor

import spacy
from spacy.lang.en import English
spacy_sent_splitter = English()
spacy_sent_splitter.add_pipe("sentencizer")
nlp = spacy.load("en_core_web_sm")

import numpy as np

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')


class SemanticSearch(AbsSearch):


        





    '''
    desc : 1. compute matching scores between query and phrase
         : 2. set_text() should be called before "search"
    query: query to be compared
    top_n: number of results 
    '''


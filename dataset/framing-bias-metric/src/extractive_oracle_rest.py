from rouge_score import rouge_scorer, scoring
from rouge import Rouge
import numpy as np
import copy
import json
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sacrebleu import corpus_bleu
import re
import nltk


# Load gold BASIL test annotations








'''
    You can refer to the greedy algorithm in 
    SummaRuNNer: A Recurrent Neural Network based Sequence Model for 
    Extractive Summarization of Documents. Here is a simple way to do it:
'''

















if __name__ == "__main__":

    
    allsides_title_triples = load_all_allsides_triples(return_type='title')
    create_title_lr_to_c_dataset(allsides_title_triples)

    # allsides_triples = load_all_allsides_triples()
    # create_left_source_2_intersection_target_dataset(allsides_triples)

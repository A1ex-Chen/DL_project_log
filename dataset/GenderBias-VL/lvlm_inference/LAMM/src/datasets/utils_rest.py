from nltk import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
stops = set(stopwords.words("english"))
import numpy as np
from nltk.corpus import wordnet as wn
import re
num_pattern =  re.compile(r'[0-9]+\.?[0-9]*')











    

















num_dict = {
    'no': 0, 'zero':0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
    'nine' : 9, 'ten':10, 'eleven':11 , 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
}


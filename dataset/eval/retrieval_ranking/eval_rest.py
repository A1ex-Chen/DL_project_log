""" Evaluate the models on Phrase Retrieval dataset
"""

import sys
import os
import time
import json
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset

from config import ROOT_DIR
from config import CreateLogger
from semsearch import SemanticSearch


sys.path.append("../../")
logger = CreateLogger()












if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--extractor', help='model name', choices=['ngrams', 'noun_chunks'])
    parser.add_argument('--ngram_min', type=int, help='ngram min', default=2)
    parser.add_argument('--ngram_max', type=int, help='ngram max', default=3)
    parser.add_argument('--scorer', help='model name', choices=['BERT', 'SentenceBERT', 'SpanBERT', 'USE', 'SimCSE', 'PhraseBERT', 'DensePhrases'])
    parser.add_argument('--scorer_type', help='transformers type',
                        choices=['bert-base-uncased', 'bert-large-uncased', 'sentence-transformers/bert-base-nli-stsb-mean-tokens',
                                 'SpanBERT/spanbert-base-cased', 'use-v5', 'princeton-nlp/sup-simcse-bert-base-uncased',
                                 'whaleloops/phrase-bert', 'princeton-nlp/densephrases-multi-query-multi',
                                 'bert-base-uncased-qa', 'bert-large-uncased-qa', 'sbert-base-nli-stsb-mean-tokens-qa',
                                 'phrase-bert-qa', 'spanbert-base-cased-qa', 'sup-simcse-bert-base-uncased-qa'], default="")

    parser.add_argument('--dataset', default="PiC/phrase_retrieval")
    parser.add_argument('--data_subset', help='subset of the dataset')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--oracle_candidates', action="store_true")
    parser.add_argument('--contextual', action="store_true")
    parser.add_argument('--context_window', type=int, help='context boundary for a phrase', default=-1)
    parser.add_argument('--max_seq_length', type=int, help='define max seq length of a sentence to handle', default=128)
    parser.add_argument('--outdir', help='output directory')

    args = parser.parse_args()

    format_string = " - {:35}: {}"
    print("{:=^70}".format(" EVALUATION CONFIGURATION "))
    print("Summary")
    print("-" * 70)
    for k, v in vars(args).items():
        print(format_string.format(k, v))

    start_time = time.time()
    sys.argv = [sys.argv[0]]    # Remove arguments to avoid crashing DensePhrases's argparse

    if args.scorer == "USE" and args.contextual:
        sys.exit('Message: USE-v5 is only supported for non-contextual phrase embeddings. Please try other models!')
    elif args.scorer == "DensePhrases":
        sys.exit("Message: DensePhrases is currently not supported. Please try other models!")

    # ThangPM: Run semantic search
    run(args)

    logger.info("elapsed time: %.2f s", time.time() - start_time)


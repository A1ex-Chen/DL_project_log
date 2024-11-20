"""

Modified from:
https://gist.github.com/SandroLuck/d04ba5c2ef710362f2641047250534b2#file-pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-py

"""

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from pytorch_lightning import seed_everything
import numpy as np
from model_path import MODEL_PATH
from cls_utils import get_data_emb
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import json
from os.path import exists
from os import makedirs






if __name__ == '__main__':
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--full_run_mode", action='store_true')
    parser.add_argument("--task", type=str, default='phrase_similarity', choices=['phrase_similarity'])
    parser.add_argument("--result_dir", type=str, default='')
    parser.add_argument("--contextual", action='store_true')

    args = parser.parse_args()
    main(args)

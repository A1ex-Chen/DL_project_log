import argparse
import deepspeed
import json
import logging
import numpy as np
import os
import random
import time
import torch
from tqdm import tqdm
from transformers.deepspeed import HfDeepSpeedConfig
import yaml

from model import load_model
from datasets import load_dataset

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"














if __name__ == "__main__":
    args = parser_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    args = vars(args)
    # arguments from command line have higher priority
    cfg.update(args)
    main(**cfg)
#!/usr/bin/env python

import argparse
import shutil
import time
from json import JSONDecodeError
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import (
    Seq2SeqDataset,
    calculate_bleu,
    calculate_rouge,
    chunks,
    lmap,
    load_json,
    parse_numeric_n_bool_cl_kwargs,
    save_json,
    use_task_specific_params,
    write_txt_file,
)


logger = getLogger(__name__)








    # Unreachable


if __name__ == "__main__":
    # Usage for MT:
    run_generate()
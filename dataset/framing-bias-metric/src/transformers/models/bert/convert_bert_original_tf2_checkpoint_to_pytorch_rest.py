"""
This script can be used to convert a head-less TF2.x Bert model to PyTorch, as published on the official GitHub:
https://github.com/tensorflow/models/tree/master/official/nlp/bert

TF2.x uses different variable names from the original BERT (TF 1.4) implementation. The script re-maps the TF2.x Bert
weight names to the original names, so the model can be imported with Huggingface/transformer.

You may adapt this script to include classification/MLM/NSP/etc. heads.
"""
import argparse
import os
import re

import tensorflow as tf
import torch

from transformers import BertConfig, BertModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow 2.x checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model (must include filename).",
    )
    args = parser.parse_args()
    convert_tf2_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
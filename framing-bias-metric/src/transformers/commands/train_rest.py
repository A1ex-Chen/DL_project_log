import os
from argparse import ArgumentParser, Namespace

from transformers import SingleSentenceClassificationProcessor as Processor
from transformers import TextClassificationPipeline, is_tf_available, is_torch_available
from transformers.commands import BaseTransformersCLICommand

from ..utils import logging


if not is_tf_available() and not is_torch_available():
    raise RuntimeError("At least one of PyTorch or TensorFlow 2.0+ should be installed to use CLI training")

# TF training parameters
USE_XLA = False
USE_AMP = False




class TrainCommand(BaseTransformersCLICommand):
    @staticmethod




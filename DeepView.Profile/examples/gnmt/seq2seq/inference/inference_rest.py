import contextlib
import logging
import os
import subprocess
import time

import torch
import torch.distributed as dist

import seq2seq.data.config as config
from seq2seq.inference.beam_search import SequenceGenerator
from seq2seq.utils import AverageMeter
from seq2seq.utils import barrier
from seq2seq.utils import get_rank
from seq2seq.utils import get_world_size




class Translator:
    """
    Translator runs validation on test dataset, executes inference, optionally
    computes BLEU score using sacrebleu.
    """





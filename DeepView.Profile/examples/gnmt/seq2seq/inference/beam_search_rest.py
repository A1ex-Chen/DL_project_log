import torch

from seq2seq.data.config import BOS
from seq2seq.data.config import EOS


class SequenceGenerator:
    """
    Generator for the autoregressive inference with beam search decoding.
    """


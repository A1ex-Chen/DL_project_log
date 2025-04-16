import itertools

import torch
import torch.nn as nn

import seq2seq.data.config as config
from seq2seq.models.attention import BahdanauAttention
from seq2seq.utils import init_lstm_


class RecurrentAttention(nn.Module):
    """
    LSTM wrapped with an attention module.
    """



class Classifier(nn.Module):
    """
    Fully-connected classifier
    """



class ResidualRecurrentDecoder(nn.Module):
    """
    Decoder with Embedding, LSTM layers, attention, residual connections and
    optinal dropout.

    Attention implemented in this module is different than the attention
    discussed in the GNMT arxiv paper. In this model the output from the first
    LSTM layer of the decoder goes into the attention module, then the
    re-weighted context is concatenated with inputs to all subsequent LSTM
    layers in the decoder at the current timestep.

    Residual connections are enabled after 3rd LSTM layer, dropout is applied
    on inputs to LSTM layers.
    """




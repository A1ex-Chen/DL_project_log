from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Hparams:
    kernel1: int = 3
    kernel2: int = 4
    kernel3: int = 5
    embed_dim: int = 300
    n_filters: int = 300
    sent_len: int = 512
    vocab_size: int = 10_000


class Conv1dPool(nn.Module):
    """Conv1d => AdaptiveMaxPool1d => Relu"""





class MultitaskClassifier(nn.Module):
    """Multi-task Classifier
    Args:
        input_dim: input dimension for each of the linear layers
        tasks: dictionary of tasks and their respective number of classes
    """





class MTCNN(nn.Module):
    """Multi-task CNN a la Yoon Kim
    Args:
        tasks: dictionary of tasks and their respective number of classes.
               This is used by the MultitaskClassifier.
        hparams: dataclass of the model hyperparameters
    """






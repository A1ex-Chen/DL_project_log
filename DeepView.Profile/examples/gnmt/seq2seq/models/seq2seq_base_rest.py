import torch.nn as nn
from torch.nn.functional import log_softmax


class Seq2Seq(nn.Module):
    """
    Generic Seq2Seq module, with an encoder and a decoder.
    """



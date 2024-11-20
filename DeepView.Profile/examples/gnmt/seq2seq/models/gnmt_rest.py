import torch.nn as nn

import seq2seq.data.config as config
from seq2seq.models.decoder import ResidualRecurrentDecoder
from seq2seq.models.encoder import ResidualRecurrentEncoder
from seq2seq.models.seq2seq_base import Seq2Seq


class GNMT(Seq2Seq):
    """
    GNMT v2 model
    """

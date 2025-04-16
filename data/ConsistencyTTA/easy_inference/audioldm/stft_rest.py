import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny, normalize, pad_center
from librosa.filters import mel as librosa_mel_fn








class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""






class TacotronSTFT(torch.nn.Module):



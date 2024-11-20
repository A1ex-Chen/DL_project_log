import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharRNN(nn.Module):

    # pass x as a pack padded sequence please.

# Code taken from:
# - https://github.com/osmr/imgclsmob

import torch
import torch.nn as nn




class ChannelShuffle(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """



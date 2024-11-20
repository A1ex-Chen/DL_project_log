import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# helper function






# top k filtering



    return inner


# top k filtering


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class AutoregressiveWrapper(nn.Module):
    """
    AutoregressiveWrapper is a wrapper class that adds autoregressive generation functionality to a given neural network.

    Args:
        net (nn.Module): The neural network model.
        max_seq_len (int): The maximum sequence length for generation. Defaults to 2048.
        pad_value (int): The padding value for generated sequences. Defaults to 0.
    """


    @torch.no_grad()
    @eval_decorator

import numpy as np
import torch
from torch import nn

from model import GPTConfig, GPT

# Batch size.
block_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# model
n_layer = 16  
n_head = 16  
n_embd = 512  
dropout = 0.0
vocab_size = 65
bias = False

# Adamw optimizer
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95


# optimizer






    return iteration
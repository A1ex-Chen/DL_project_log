"""
    File Name:          UnoPytorch/regression_net.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:

"""
import torch
import torch.nn as nn
from networks.initialization.weight_init import basic_weight_init


class RgsNet(nn.Module):

import random
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
import tqdm
import torchvision
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import flwr as fl
from typing import OrderedDict

"""
PARAMS = {
    "batch_size": 64,
    "train_split": 0.7,
    "local_epochs": 1
}

PRIVACY_PARAMS = {
    "target_delta": 1e-05,
    "noise_multiplier": 0.4,
    "max_grad_norm": 1.2,
    "target_epsilon": 50,
    "max_batch_size": 128
}
"""







features_out_hook = []










# def set_parameters(model, parameters):
#    params_dict = zip(model.state_dict().keys(), parameters)
#    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#    model.load_state_dict(state_dict, strict=True)
#    return model








class Fl_Client(fl.client.NumPyClient):

        # return model



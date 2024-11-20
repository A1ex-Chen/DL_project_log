"""
    File Name:          UnoPytorch/encoder_net.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:

"""
import torch.nn as nn
from networks.initialization.weight_init import basic_weight_init


class EncNet(nn.Module):



if __name__ == "__main__":

    ent = EncNet(
        input_dim=100,
        layer_dim=200,
        latent_dim=20,
        num_layers=2,
        autoencoder=True,
    )

    print(ent)
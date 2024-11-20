import numpy as np
import time
import torch
import torch.nn as nn

















    list_conv1d = []


    list_linear = []


    list_bn = []


    list_relu = []


    list_pooling2d = []


    list_pooling1d = []



    # Register hook
    foo(model)

    device = device = next(model.parameters()).device
    input = torch.rand(1, audio_length).to(device)

    out = model(input)

    total_flops = (
        sum(list_conv2d)
        + sum(list_conv1d)
        + sum(list_linear)
        + sum(list_bn)
        + sum(list_relu)
        + sum(list_pooling2d)
        + sum(list_pooling1d)
    )

    return total_flops
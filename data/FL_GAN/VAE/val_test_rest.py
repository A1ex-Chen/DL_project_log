import argparse

import torch
import torchvision.models

from vq_vae_gated_pixelcnn_prior import train_vq_vae_with_gated_pixelcnn_prior
from utils import load_data
import numpy as np
import os
import math
import opacus



    # return metrics




if __name__ == "__main__":
    args = args_function()
    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client

    trainset, testset = load_data(DATASET)

    print(f"Dataset:{DATASET}")
    print(f"Client: {CLIENT}")
    print(f"Device:{args.device}")
    print(f"Differential Privacy: {DP}")

    # model = resnet = torchvision.models.resnet18(num_classes=10)
    # model = opacus.validators.ModuleValidator.fix(model)

    main(args, trainset, testset)
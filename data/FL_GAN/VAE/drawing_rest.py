"""
load the weights of models and draw figures
latent space
FID
Sampling and reconstruction
"""
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torchvision
from utils import calculate_fid, load_data
from torch.utils.data import DataLoader
import tqdm
import os
import opacus.validators
from opacus import PrivacyEngine
import warnings

from VAE_Torch.vq_vae_gated_pixelcnn_prior import Improved_VQVAE

warnings.filterwarnings("ignore")

features_out_hook = []







if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Draw figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--grad_sample_mode", type=str, default="hooks")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar",
        help="Dataset",
    )

    parser.add_argument(
        "--dp",
        type=str,
        default="normal",
        help="Disable privacy training and just train with vanilla type",
    )

    parser.add_argument(
        "--client",
        type=int,
        default=1,
        help="Number of clients, 1 for centralized, 2/3/4/5 for federated learning",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="default GPU ID for model",
    )

    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to load",
    )

    parser.add_argument(
        "--secure_rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )

    parser.add_argument(
        "-c",
        "--max_per_sample_grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )

    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )

    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for adam. default=0.999"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-05,
        metavar="WD",
        help="weight decay",
    )

    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )

    parser.add_argument(
        "-D",
        "--embedding_dim",
        type=int,
        default=256,  # 64, 256
        help="Embedding dimention"
    )

    parser.add_argument(
        "-K",
        "--num_embeddings",
        type=int,
        default=512,  # 512, 128
        help="Embedding dimention"
    )

    args = parser.parse_args()

    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client
    DEVICE = args.device

    _, testset = load_data(DATASET)
    print(DEVICE)
    # model = VAE(DATASET, DEVICE)
    # model = VQVAE(3, 64, 512)

    # model.to(DEVICE)
    testLoader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    test_images, test_labels = next(iter(testLoader))

    """
    if args.dp == "gaussian":
        model = opacus.validators.ModuleValidator.fix(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                     betas=(args.beta1, args.beta2))

        
        if args.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                                args.max_per_sample_grad_norm / np.sqrt(n_layers)
                            ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm
        
        max_grad_norm = args.max_per_sample_grad_norm
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        # clipping = "per_layer" if args.clip_per_layer else "flat"
        model, optimizer, testLoader = privacy_engine.make_private(
            module=model,
            data_loader=testLoader,
            optimizer=optimizer,
            noise_multiplier=args.sigma,
            grad_sample_mode=args.grad_sample_mode,
            max_grad_norm=max_grad_norm
            #clipping=clipping
        )
    """

    with torch.no_grad():
        results = validation(args, test_images, test_labels)
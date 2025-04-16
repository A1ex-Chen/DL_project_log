import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from VAE_Torch.vq_vae import VQVAE
from utils import get_labels
import tqdm
from opacus import PrivacyEngine
from opacus.validators import register_module_validator

import opacus
import os
import matplotlib.pyplot as plt





# L2 Distance squared
l2_dist = lambda x, y: (x - y) ** 2





    # return {**vq_vqe_metrics, **pixel_cnn_metrics}


    # Training
    # Prior_only showing choose which model to process: false->vqvaq, true->gatedpixel cnn


# Loss

            sampling_images = sample(args.num_sampling)
            sample_images_save_path = sampling_save_path + f"/sampling_images_at_epoch_{epoch+1:03d}.png"
            imshow(torchvision.utils.make_grid(sampling_images, nrow=6), sample_images_save_path)

            print(
                f"[{100*(epoch+1)/no_epochs:.2f}%] Epoch {epoch + 1} - "
                f"Train loss: {np.mean(train_losses):.2f} - "
                f"Validate loss: {validate_loss:.2f} - "
                f"Test loss: {test_loss:.2f} - "
                f"Time elapsed: {time.time() - epoch_start:.2f}", end=""
            )
        if args.dp == "gaussian":
            epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
            EPSILON.append(epsilon.item())
            print(f"(ε = {epsilon:.2f}, δ = {args.delta})")
        else:
            print("\n")

        # Save VQ-VAE model with GatedPixelCNN
        if prior_only:
            # Save minimum test loss of the model when after trained VQ VAE
            if test_loss < min_loss:
                min_loss = test_loss
                weight_save_pth = f"Results/{DATASET}/{CLIENT}/{DP}/weights"
                if not os.path.exists(weight_save_pth):
                    os.makedirs(weight_save_pth)
                weight_save_pth += f"/weights_central_{args.epochs}.pt"
                torch.save(model.state_dict(), weight_save_pth)

        if not prior_only:
            metrics = {
                "vq_vae_train_losses": np.array(train_losses),
                "vq_vae_validate_losses": np.array(validate_losses),
                "vq_vae_test_losses": np.array(test_losses)
            }
        else:
            metrics = {
                "pixcel_cnn_train_losses": np.array(train_losses),
                "pixcel_cnn_validate_losses": np.array(validate_losses),
                "pixcel_cnn_test_losses": np.array(test_losses)
            }

        if DP == "gaussian":
            metrics["epsilon"] = np.array(EPSILON)
        return metrics


# Loss
def get_batched_loss(args, data_loader, model, loss_func, loss_triples=True):
    """
    Gets loss in a batched fashion.
    Input is data loader, model to produce output and loss function
    Assuming loss output is VLB, reconstruct_loss, KL
    """
    losses = [[], [], []] if loss_triples else []  # [VLB, Reconstruction Loss, KL] or [Loss]
    loop = tqdm.tqdm((data_loader), total=len(data_loader), leave=False)
    model.eval()
    for images, labels in loop:
        if loop.last_print_n > 0:
            break

        images = images.to(args.device)
        out = model(images)
        loss = loss_func(images, out)

        if loss_triples:
            losses[0].append(loss[0].cpu().item())
            losses[1].append(loss[1].cpu().item())
            losses[2].append(loss[2].cpu().item())
        else:
            losses.append(loss.cpu().item())

    losses = np.array(losses)

    if not loss_triples:
        return np.mean(losses)
    return np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2])
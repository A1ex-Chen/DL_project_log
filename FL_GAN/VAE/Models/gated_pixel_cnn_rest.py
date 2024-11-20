import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class MaskedConv2d(nn.Conv2d):
    """
    Implements a conv2d with mask applied on its weights.

    Args:
        mask (torch.Tensor): the mask tensor.
        in_channels (int) – Number of channels in the input image.
        out_channels (int) – Number of channels produced by the convolution.
        kernel_size (int or tuple) – Size of the convolving kernel
    """




class VerticalStackConv(MaskedConv2d):



class HorizontalStackConv(MaskedConv2d):


class GatedMaskedConv(nn.Module):




# GatedPixelCNN

class GatedPixelCNN(nn.Module):




if __name__ == "__main__":
    #data_transform = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #testset = MNIST(root="../data", train=False, transform=data_transform, download=True)
    model = GatedPixelCNN(1, 128, 1)
    #testLoader = DataLoader(testset, batch_size=128, shuffle=False)
    # val_images, val_labels = next(iter(testLoader))
    x = torch.Tensor(numpy.random.rand(64, 1, 7, 7))
    result = model(x)
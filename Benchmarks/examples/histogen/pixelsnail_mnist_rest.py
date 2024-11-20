import numpy as np
import torch
from pixelsnail import PixelSNAIL
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm




class PixelTransform:



if __name__ == "__main__":
    device = "cuda"
    epoch = 10

    dataset = datasets.MNIST(".", transform=PixelTransform(), download=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = PixelSNAIL([28, 28], 256, 128, 5, 2, 4, 128)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(10):
        train(i, loader, model, optimizer, device)
        torch.save(model.state_dict(), f"checkpoint/mnist_{str(i + 1).zfill(3)}.pt")
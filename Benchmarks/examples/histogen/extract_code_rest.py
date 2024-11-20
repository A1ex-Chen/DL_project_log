import argparse
import pickle

import lmdb
import torch
from dataset import CodeRow, ImageFileDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from vqvae import VQVAE




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    device = "cuda"

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFileDataset(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)
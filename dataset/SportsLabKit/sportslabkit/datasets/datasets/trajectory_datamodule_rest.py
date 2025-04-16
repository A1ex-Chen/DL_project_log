from functools import partial
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose








class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, transform=None, flatten=False, split=50):
        self.flatten = flatten
        self.transform = transform
        self.files = self.get_files(data_dir)
        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.load_data(self.files[idx])
        if not self.flatten:
            data = data.reshape(data.shape[0], -1, 2)  # (seq_len, num_agents, 2)

        if self.transform:
            data = self.transform(data)

        out_data = data[: self.split]
        out_label = data[self.split :]
        return out_data, out_label

    def load_data(self, path):
        data = np.loadtxt(path, delimiter=",")
        return data

    def get_files(self, data_dir):
        files = []
        for file in data_dir.glob("*.txt"):
            files.append(file)
        return files










def random_ordering(data):
    # randomize and flatten the agent axis
    num_agents = data.shape[1]
    data = data[:, torch.randperm(num_agents), :]
    return data


def smooth(data):
    return data


class TrajectoryDataModule(pl.LightningDataModule):





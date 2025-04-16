import os
import pickle
from collections import namedtuple

import lmdb
import torch
from torch.utils.data import Dataset
from torchvision import datasets

CodeRow = namedtuple("CodeRow", ["top", "bottom", "filename"])


class ImageFileDataset(datasets.ImageFolder):


class LMDBDataset(Dataset):


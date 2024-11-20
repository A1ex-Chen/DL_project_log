from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from collections.abc import Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from sportslabkit.image_model.base import BaseImageModel
from sportslabkit.image_model.clip import CLIP_RN50
from sportslabkit.image_model.torchreid import ShuffleNet
from sportslabkit.image_model.visualization import plot_tsne
from sportslabkit.logger import logger
from sportslabkit.types.detection import Detection


__all__ = [
    "ShuffleNet",
    "CLIP_RN50",
    "plot_tsne",
    "show_torchreid_models",
]






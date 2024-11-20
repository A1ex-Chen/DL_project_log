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

from sportslabkit.vector_model.base import BaseVectorModel
from sportslabkit.vector_model.sklearn import SklearnVectorModel
from sportslabkit.logger import logger
from sportslabkit.types.detection import Detection






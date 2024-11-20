import logging

import candle
import darts
import example_setup as bmk
import torch
import torch.nn as nn
from operations import OPS, Stem
from torch import optim
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("darts_advanced")














if __name__ == "__main__":
    main()
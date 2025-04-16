import logging
import sys

import candle
import darts
import example_setup as bmk
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# logging.basicConfig(sys.stdout, level=logging.INFO)
# Set up the logger to go to stdout instead of stderr
logger = logging.getLogger("darts_uno")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))












if __name__ == "__main__":
    main()
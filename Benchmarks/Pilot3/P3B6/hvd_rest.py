import argparse

import horovod.torch as hvd
import numpy as np
import torch
import torch.nn as nn
from bert import HiBERT
from random_data import MimicDatasetSynthetic
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# import candle
# import p3b6 as bmk


hvd.init()


















if __name__ == "__main__":
    main()
from pathlib import Path

import candle
import p3b7 as bmk
import pandas as pd
import torch
from data import P3B3, Egress
from meters import AccuracyMeter
from metrics import F1Meter
from mtcnn import MTCNN, Hparams
from prune import create_prune_masks, remove_prune_masks
from torch.utils.data import DataLoader
from util import to_device

TASKS = {
    "subsite": 15,
    "laterality": 3,
    "behavior": 3,
    "grade": 3,
}

TRAIN_F1_MICRO = F1Meter(TASKS, "micro")
VALID_F1_MICRO = F1Meter(TASKS, "micro")

TRAIN_F1_MACRO = F1Meter(TASKS, "macro")
VALID_F1_MACRO = F1Meter(TASKS, "macro")




















if __name__ == "__main__":
    main()
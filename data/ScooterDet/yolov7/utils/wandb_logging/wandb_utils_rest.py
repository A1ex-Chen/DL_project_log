import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))  # add utils/ to path
from utils.datasets import LoadImagesAndLabels
from utils.datasets import img2label_paths
from utils.general import colorstr, xywh2xyxy, check_dataset

try:
    import wandb
    from wandb import init, finish
except ImportError:
    wandb = None

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'












class WandbLogger():












import sys
from pathlib import Path

import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from train import parse_opt, train
from utils.callbacks import Callbacks
from utils.general import increment_path
from utils.torch_utils import select_device




if __name__ == "__main__":
    sweep()
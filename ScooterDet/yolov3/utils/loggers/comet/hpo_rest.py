import argparse
import json
import logging
import os
import sys
from pathlib import Path

import comet_ml

logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from train import train
from utils.callbacks import Callbacks
from utils.general import increment_path
from utils.torch_utils import select_device

# Project Configuration
config = comet_ml.config.get_config()
COMET_PROJECT_NAME = config.get_string(os.getenv("COMET_PROJECT_NAME"), "comet.project_name", default="yolov5")






if __name__ == "__main__":
    opt = get_args(known=True)

    opt.weights = str(opt.weights)
    opt.cfg = str(opt.cfg)
    opt.data = str(opt.data)
    opt.project = str(opt.project)

    optimizer_id = os.getenv("COMET_OPTIMIZER_ID")
    if optimizer_id is None:
        with open(opt.comet_optimizer_config) as f:
            optimizer_config = json.load(f)
        optimizer = comet_ml.Optimizer(optimizer_config)
    else:
        optimizer = comet_ml.Optimizer(optimizer_id)

    opt.comet_optimizer_id = optimizer.id
    status = optimizer.status()

    opt.comet_optimizer_objective = status["spec"]["objective"]
    opt.comet_optimizer_metric = status["spec"]["metric"]

    logger.info("COMET INFO: Starting Hyperparameter Sweep")
    for parameter in optimizer.get_parameters():
        run(parameter["parameters"], opt)
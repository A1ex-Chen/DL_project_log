# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner






DetectionTrainer.get_model = patched_get_model
v8DetectionLoss.__init__ = patched_loss_init
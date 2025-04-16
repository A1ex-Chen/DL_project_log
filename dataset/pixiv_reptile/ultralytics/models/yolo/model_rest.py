# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import ROOT, yaml_load


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""


    @property


class YOLOWorld(Model):
    """YOLO-World object detection model."""


    @property

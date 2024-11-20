import inspect

from sportslabkit.detection_model.base import BaseDetectionModel
from sportslabkit.detection_model.dummy import DummyDetectionModel
from sportslabkit.detection_model.yolov8 import YOLOv8, YOLOv8l, YOLOv8m, YOLOv8n, YOLOv8s, YOLOv8x
from sportslabkit.logger import logger


__all__ = [
    "BaseDetectionModel",
    "load",
    "show_available_models",
    "YOLOv8",
    "YOLOv8n",
    "YOLOv8s",
    "YOLOv8m",
    "YOLOv8l",
    "YOLOv8x",
    "DummyDetectionModel",
]








if __name__ == "__main__":
    for cls in inheritors(BaseDetectionModel):
        print(cls.__name__)
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.backbone.resnet import resnet
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.backbone.yolo import YOLOv5Backbone, YOLOv8Backbone


__all__ = ['build_backbone']

BACKBONE_MAP = {
    'resnet': resnet,
    'yolo5': YOLOv5Backbone,
    'yolo8': YOLOv8Backbone,
}


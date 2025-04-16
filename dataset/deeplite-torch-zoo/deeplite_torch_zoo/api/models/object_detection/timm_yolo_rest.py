from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.timm_model import TimmYOLO
from deeplite_torch_zoo.api.models.object_detection.helpers import make_wrapper_func, load_pretrained_model
from deeplite_torch_zoo.api.models.object_detection.timm_yolo_backbones import SUPPORTED_BACKBONES
from deeplite_torch_zoo.api.datasets.object_detection.yolo import DATASET_CONFIGS

__all__ = []




for dataset_tag, dataset_config in DATASET_CONFIGS.items():
    for backbone_tag in SUPPORTED_BACKBONES:
        model_tag = f'yolo_timm_{backbone_tag}'
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(yolo_timm, name, model_tag, dataset_tag,
                                            dataset_config.num_classes)
        __all__.append(name)
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6_model import YOLOv6
from deeplite_torch_zoo.api.models.object_detection.helpers import (
    make_wrapper_func, get_project_root, load_pretrained_model
)
from deeplite_torch_zoo.api.datasets.object_detection.yolo import DATASET_CONFIGS


__all__ = []


CFG_PATH = 'deeplite_torch_zoo/src/object_detection/yolo/flexible_yolo/yolov6/configs'

YOLOV6_CONFIGS = {
    'yolo6s': 'yolov6s.py',
    'yolo6m': 'yolov6m.py',
    'yolo6l': 'yolov6l.py',
    'yolo6s_lite_s': 'yolov6_lite_s.py',
    'yolo6s_lite_m': 'yolov6_lite_m.py',
    'yolo6s_lite_l': 'yolov6_lite_l.py',
}

DEFAULT_MODEL_SCALES = {
    # [depth, width]
    'd33w25': [0.33, 0.25],
    'd33w5': [0.33, 0.50],
    'd6w75': [0.6, 0.75],
    'd1w1': [1.00, 1.00],
    'd1w5': [1.0, 0.5],
    'd1w25': [1.0, 0.25],
    'd1w75': [1.0, 0.75],
    'd33w1': [0.33, 1.0],
    'd33w75': [0.33, 0.75],
    'd6w1': [0.6, 1.0],
    'd6w5': [0.6, 0.5],
    'd6w25': [0.6, 0.25],
}






full_model_dict = {}
for model_key, config_name in YOLOV6_CONFIGS.items():
    for cfg_name, param_dict in get_model_scales().items():
        full_model_dict[f'{model_key}-{cfg_name}'] = {
            'params': param_dict,
            'config': get_project_root() / CFG_PATH / config_name,
        }


for dataset_tag, dataset_config in DATASET_CONFIGS.items():
    for model_tag, model_dict in full_model_dict.items():
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(
            yolov6,
            name,
            model_tag,
            dataset_tag,
            dataset_config.num_classes,
            config_path=model_dict['config'],
            **model_dict['params'],
        )
        __all__.append(name)
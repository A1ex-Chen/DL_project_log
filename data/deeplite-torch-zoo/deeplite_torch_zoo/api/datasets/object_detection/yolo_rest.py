# Ultralytics YOLO 🚀, AGPL-3.0 license

import pathlib
from collections import namedtuple
from functools import partial

from addict import Dict

from deeplite_torch_zoo.utils import RANK
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY

from deeplite_torch_zoo.src.object_detection.datasets.dataloader import get_dataloader
from deeplite_torch_zoo.api.datasets.object_detection.utils import check_det_dataset


__all__ = []

HERE = pathlib.Path(__file__).parent

CONFIG_VARS = ['rect', 'cache', 'single_cls', 'task', 'classes', 'fraction',
               'mosaic', 'mixup', 'mask_ratio', 'overlap_mask', 'copy_paste', 'degrees',
               'translate', 'scale', 'shear', 'perspective', 'fliplr', 'flipud', 'hsv_h', 'hsv_s', 'hsv_v']

DEFAULT_IMG_RES = 640
DatasetConfig = namedtuple('DatasetConfig', ['yaml_file', 'num_classes', 'default_res'])
DATASET_CONFIGS = {
    'voc': DatasetConfig('VOC.yaml', 20, 448),
    'coco': DatasetConfig('coco.yaml', 80, 640),
    'coco8': DatasetConfig('coco8.yaml', 80, 640),
    'coco128': DatasetConfig('coco128.yaml', 80, 640),
    'SKU-110K': DatasetConfig('SKU-110K.yaml', 1, 640),
    'person_detection': DatasetConfig('coco_person.yaml', 1, 640),
}




for dataset_name_key in DATASET_CONFIGS:
    wrapper_fn_name = f'get_{dataset_name_key}_for_yolo'
    wrapper_fn = partial(create_detection_dataloaders, dataset_config=dataset_name_key)
    globals()[wrapper_fn_name] = wrapper_fn
    DATASET_WRAPPER_REGISTRY.register(dataset_name=dataset_name_key)(wrapper_fn)
    __all__.append(wrapper_fn_name)
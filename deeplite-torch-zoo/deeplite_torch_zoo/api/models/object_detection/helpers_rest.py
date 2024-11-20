from pathlib import Path
import urllib.parse as urlparse

import deeplite_torch_zoo
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.models.object_detection.checkpoints import (
    CHECKPOINT_STORAGE_URL,
    model_urls,
)
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


DATASET_LIST = [
    ('voc', 20),
    ('coco', 80),
    ('coco128', 80),
    ('coco8', 80),
    ('SKU-110K', 1),
]







    wrapper_func.__name__ = wrapper_name
    return wrapper_func
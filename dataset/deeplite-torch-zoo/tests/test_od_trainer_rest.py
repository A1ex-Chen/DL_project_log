import pytest

from deeplite_torch_zoo import get_model
from deeplite_torch_zoo.trainer import Detector

TEST_MODELS = [
    ('yolo7n', ),
    ('yolo_fdresnet18x0.25', ),
    ('yolo_timm_tinynet_e', ),
    ('yolo6s-d33w25', ),
    ('yolonas_s', ),
]


@pytest.mark.parametrize(
    ('model_name', ),
    TEST_MODELS,
)


@pytest.mark.parametrize(
    ('model_name', ),
    TEST_MODELS,
)
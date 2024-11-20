import pytest

from deeplite_torch_zoo import (get_dataloaders, get_eval_function,
                                get_model)


BATCH_SIZE = 128

TEST_LOADER_KEY_MAP = {
    'tinyimagenet': 'val',
    'vww': 'test',
}

ACC_KEY_MAP = {
    'tinyimagenet': 'acc',
    'vww': 'acc',
}

DATASET_PATHS = {
    'tinyimagenet': '/neutrino/datasets/TinyImageNet/',
    'vww': '/neutrino/datasets/vww',
}

@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'reference_accuracy'),
    [
        ('vgg19', 'tinyimagenet', 0.728),
        ('mobilenet_v2', 'tinyimagenet', 0.680),
        ('resnet18', 'tinyimagenet', 0.663),
        ('resnet34', 'tinyimagenet', 0.686),
        ('resnet50', 'tinyimagenet', 0.730),
        ('mobilenetv3_large', 'vww', 0.891),
        ('mobilenetv3_small', 'vww', 0.892),
    ]
)
@pytest.mark.slow
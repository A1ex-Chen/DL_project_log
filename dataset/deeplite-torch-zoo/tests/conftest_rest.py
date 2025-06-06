import shutil
import contextlib
from pathlib import Path

import pytest
import torch

from deeplite_torch_zoo import get_dataloaders


@pytest.fixture


@pytest.fixture


@pytest.fixture

    yield set_torch_seed


@pytest.fixture
def imagewoof160_dataloaders(data_root='./', batch_size=32):
    p = Path(data_root)
    dataloaders = get_dataloaders(
        dataset_name='imagewoof_160',
        data_root=data_root,
        batch_size=batch_size,
        map_to_imagenet_labels=True,
        device='cpu',
    )
    yield dataloaders
    (p / 'imagewoof160.zip').unlink()
    shutil.rmtree(p / 'imagewoof160')


@pytest.fixture
def cifar100_dataloaders(data_root='./', batch_size=32):
    p = Path(data_root)
    dataloaders = get_dataloaders(
        dataset_name='cifar100',
        data_root=data_root,
        batch_size=batch_size,
        device='cpu',
    )
    yield dataloaders
    (p / 'cifar-100-python.tar.gz').unlink()
    shutil.rmtree(p / 'cifar-100-python')
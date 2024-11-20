import shutil
from pathlib import Path

import pytest

from deeplite_torch_zoo import get_dataloaders


DATASETS_ROOT = Path('/neutrino/datasets/')
BATCH_SIZE = 128


@pytest.mark.parametrize(
    ('dataset_name', 'tmp_dataset_files', 'tmp_dataset_folders',
     'train_dataloader_len', 'test_dataloader_len'),
    [
        ('cifar100', ('cifar-100-python.tar.gz', ), ('cifar-100-python', ), 390, 79),
        ('cifar10', ('cifar-10-python.tar.gz', ), ('cifar-10-batches-py', ), 390, 79),
        ('imagenette', ('imagenette.zip', ), ('imagenette', ), 73, 31),
        ('imagewoof', ('imagewoof.zip', ), ('imagewoof', ), 70, 31),
        ('mnist', (), ('MNIST', ), 468, 79),
        ('coco128', (), (), 1, 1),
    ],
)


@pytest.mark.parametrize(
    ('dataset_name', 'data_root', 'train_dataloader_len', 'test_dataloader_len'),
    [
        ('vww', str(DATASETS_ROOT / 'vww'), 901, 63),
        ('imagenet', str(DATASETS_ROOT / 'imagenet16'), 1408, 332),
        ('imagenet', str(DATASETS_ROOT / 'imagenet10'), 3010, 118),
        ('imagenet', str(DATASETS_ROOT / 'imagenet'), 10010, 391),
        ('coco', str(DATASETS_ROOT / 'coco'), 11829, 500),
    ],
)
@pytest.mark.local
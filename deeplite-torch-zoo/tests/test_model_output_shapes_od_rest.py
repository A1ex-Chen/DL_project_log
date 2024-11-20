import pytest

from deeplite_torch_zoo import get_dataloaders, get_model, list_models_by_dataset
from deeplite_torch_zoo.api.models.object_detection.yolo import YOLO_CONFIGS


TEST_BATCH_SIZE = 4
TEST_NUM_CLASSES = 42
COCO_NUM_CLASSES = 80

DETECTION_MODEL_TESTS = []

for model_key in YOLO_CONFIGS:
    DETECTION_MODEL_TESTS.append((f'{model_key}t', 'coco8', {'image_size': 480},
        [(3, 64, 64), (3, 32, 32), (3, 16, 16)], False, False))

for model_name in list_models_by_dataset('coco', with_checkpoint=True):
    DETECTION_MODEL_TESTS.append((model_name, 'coco8', {'image_size': 480},
        [(3, 64, 64), (3, 32, 32), (3, 16, 16)], True, True))


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'dataloader_kwargs',
     'output_shapes', 'download_checkpoint', 'check_shape'),
    DETECTION_MODEL_TESTS
)


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'dataloader_kwargs',
     'output_shapes', 'download_checkpoint', 'check_shape'),
    DETECTION_MODEL_TESTS
)
import pytest
import torch

from deeplite_torch_zoo import get_model, list_models_by_dataset

TEST_BATCH_SIZE = 2
TEST_NUM_CLASSES = 42

FLOWERS_MODEL_TESTS = []
for model_name in list_models_by_dataset('flowers102'):
    FLOWERS_MODEL_TESTS.append((model_name, 'flowers102', 224, 3, 102))


@pytest.mark.slow
@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    FLOWERS_MODEL_TESTS,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    FLOWERS_MODEL_TESTS,
)
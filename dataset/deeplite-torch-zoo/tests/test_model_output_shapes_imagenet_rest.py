import pytest
import torch

from deeplite_torch_zoo import get_model, list_models_by_dataset
from deeplite_torch_zoo.api.models.classification.model_implementation_dict import (
    FIXED_SIZE_INPUT_MODELS, INPLACE_ABN_MODELS)

TEST_BATCH_SIZE = 2
TEST_NUM_CLASSES = 42

IMAGENET_MODEL_TESTS = []
for model_name in list_models_by_dataset('imagenet'):
    if model_name not in FIXED_SIZE_INPUT_MODELS and model_name not in INPLACE_ABN_MODELS:
        IMAGENET_MODEL_TESTS.append((model_name, 'imagenet', 224, 3, 1000))


@pytest.mark.slow
@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    IMAGENET_MODEL_TESTS,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    IMAGENET_MODEL_TESTS,
)
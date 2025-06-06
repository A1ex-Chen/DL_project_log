from collections import namedtuple

import pytest
import torch

from deeplite_torch_zoo import get_model, list_models_by_dataset

Dataset = namedtuple(typename='Dataset', field_names=('name', 'img_res', 'in_channels', 'num_classes'))

TEST_BATCH_SIZE = 2
TEST_NUM_CLASSES = 42

OVERRIDE_TEST_PARAMS = {
    'vww': {'mobilenet_v1_0.25_96px': {'img_res': 96}}
}

DATASETS = [
    Dataset('tinyimagenet', 64, 3, 100),
    Dataset('imagenet16', 224, 3, 16),
    Dataset('imagenet10', 224, 3, 10),
    Dataset('vww', 224, 3, 2),
    Dataset('cifar10', 32, 3, 10),
    Dataset('cifar100', 32, 3, 100),
    Dataset('mnist', 28, 1, 10),
]

CLASSIFICATION_MODEL_TESTS = []
for dataset in DATASETS:
    for model_name in list_models_by_dataset(dataset.name):
        test_params = {
            'model_name': model_name,
            'dataset_name': dataset.name,
            'img_res': dataset.img_res,
            'in_channels': dataset.in_channels,
            'num_classes': dataset.num_classes,
        }
        if dataset.name in OVERRIDE_TEST_PARAMS and model_name in OVERRIDE_TEST_PARAMS[dataset.name]:
            test_params.update(OVERRIDE_TEST_PARAMS[dataset.name][model_name])
        CLASSIFICATION_MODEL_TESTS.append(tuple(test_params.values()))


IMAGENET_MODEL_NAMES = [
    # torchvision:
    'mobilenet_v3_small',
    'squeezenet1_0',
    # timm:
    'hrnet_w18_small_v2',
    'efficientnet_es_pruned',
    # pytorchcv:
    'fdmobilenet_wd4',
    'proxylessnas_mobile',
    # zoo:
    'mobilenetv2_w035',
    'mobileone_s0_zoo',
    'mobileone_s4_zoo',
    'fasternet_t2',
    'fasternet_s',
    'edgevit_xxs',
    # mmpretrain:
    'hornet-base_3rdparty_in1k_mmpretrain',
    'edgenext-base_3rdparty-usi_in1k_mmpretrain',
]

NO_PRETRAINED_WEIGHTS = [
    'fasternet_t2',
    'fasternet_s',
    'edgevit_xxs',
]

for model_name in IMAGENET_MODEL_NAMES:
    CLASSIFICATION_MODEL_TESTS.append((model_name, 'imagenet', 224, 3, 1000))


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    CLASSIFICATION_MODEL_TESTS,
)


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    CLASSIFICATION_MODEL_TESTS,
)
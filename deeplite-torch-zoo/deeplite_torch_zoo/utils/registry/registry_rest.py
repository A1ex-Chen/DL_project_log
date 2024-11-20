# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Registry and RegistryStorage implementation modified from:
# https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/openvino/tools/pot/utils/registry.py
#

from collections import namedtuple


class Registry:



    @property

    @property



class ModelWrapperRegistry(Registry):

    @property

    @property





class DatasetWrapperRegistry(Registry):




class EvaluatorWrapperRegistry(Registry):




class RegistryStorage:



        return wrap

    def get(self, name):
        if name not in self._registry_dict:
            raise KeyError(f'{name} was not found in the {self._name} registry')
        return self._registry_dict[name]

    @property
    def registry_dict(self):
        return self._registry_dict

    @property
    def name(self):
        return self._name

    def __add__(self, other_registry):
        """
        Adding two objects of type Registry results in merging registry dicts
        """
        res = type(self)(self._name)
        if self._registry_dict.keys() & other_registry.registry_dict.keys():
            raise ValueError('Trying to add two registries with overlapping keys')
        res._registry_dict.update(self._registry_dict)
        res._registry_dict.update(other_registry.registry_dict)
        return res


class ModelWrapperRegistry(Registry):
    def __init__(self):
        super().__init__()
        self._task_type_map = {}
        self._registry_key = namedtuple('RegistryKey', ['model_name', 'dataset_name'])
        self._registry_pretrained_models = {}

    @property
    def task_type_map(self):
        return self._task_type_map

    @property
    def pretrained_models(self):
        return self._registry_pretrained_models

    def register(self, model_name, dataset_name, task_type, has_checkpoint=True):  # pylint: disable=arguments-renamed


        return wrap

    def get(self, model_name, dataset_name):
        key = self._registry_key(model_name=model_name, dataset_name=dataset_name)
        if key not in self._registry_dict:
            raise KeyError(
                f'Model {model_name} on dataset {dataset_name} was not found '
                'in the model wrapper registry'
            )
        return self._registry_dict[key]

    def get_task_type(self, model_name, dataset_name):
        GENERIC_DATASET_TASK_TYPE_MAP = {
            'voc_format_dataset': 'voc',
        }
        if dataset_name in GENERIC_DATASET_TASK_TYPE_MAP:
            dataset_name = GENERIC_DATASET_TASK_TYPE_MAP.get(dataset_name)
        key = self._registry_key(model_name=model_name, dataset_name=dataset_name)
        if key not in self._registry_dict:
            raise KeyError(
                f'Model {model_name} on dataset {dataset_name} was not found '
                'in the model wrapper registry'
            )
        return self._task_type_map[key]


class DatasetWrapperRegistry(Registry):
    def __init__(self):
        super().__init__()
        self._task_type_map = {}
        self._registry_key = namedtuple('RegistryKey', ['dataset_name'])

    def register(self, dataset_name):  # pylint: disable=arguments-renamed

        return wrap

    def get(self, dataset_name):  # pylint: disable=arguments-renamed
        key = self._registry_key(dataset_name=dataset_name)
        if key not in self._registry_dict:
            registered_dataset_names = [key.dataset_name for key in self.registry_dict]
            raise KeyError(
                f'Dataset {dataset_name} was not found in the dataset registry. '
                f'Registered datasets: {registered_dataset_names}'
            )
        return self._registry_dict[key]


class EvaluatorWrapperRegistry(Registry):
    def __init__(self):
        super().__init__()
        self._task_type_map = {}
        self._registry_key = namedtuple(
            'RegistryKey', ['dataset_type', 'model_type', 'task_type']
        )

    def register(self, task_type, model_type=None, dataset_type=None):  # pylint: disable=arguments-renamed


        return wrap

    def get(self, task_type, model_name, dataset_name):
        key = self._registry_key(
            task_type=task_type, dataset_type=dataset_name, model_type=model_name
        )
        if key not in self._registry_dict:
            key = self._registry_key(
                task_type=task_type, model_type=model_name, dataset_type=None
            )
        if key not in self._registry_dict:
            key = self._registry_key(
                task_type=task_type, model_type=None, dataset_type=None
            )

        if key not in self._registry_dict:
            raise KeyError(
                f'{task_type} model {model_name} on dataset {dataset_name} was not found '
                'in the evaluator wrapper registry'
            )

        return self._registry_dict[key]


class RegistryStorage:
    def __init__(self, registry_list):
        regs = [r for r in registry_list if isinstance(r, Registry)]
        self.registries = {}
        for r in regs:
            if r.name in self.registries:
                raise RuntimeError(
                    f'There are more than one registry with the name "{r.name}"'
                )
            self.registries[r.name] = r

    def get(self, registry_name: str) -> Registry:
        if registry_name not in self.registries:
            raise RuntimeError(
                f'Cannot find registry with registry_name "{registry_name}"'
            )
        return self.registries[registry_name]
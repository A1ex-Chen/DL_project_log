def get_task_type(self, model_name, dataset_name):
    GENERIC_DATASET_TASK_TYPE_MAP = {'voc_format_dataset': 'voc'}
    if dataset_name in GENERIC_DATASET_TASK_TYPE_MAP:
        dataset_name = GENERIC_DATASET_TASK_TYPE_MAP.get(dataset_name)
    key = self._registry_key(model_name=model_name, dataset_name=dataset_name)
    if key not in self._registry_dict:
        raise KeyError(
            f'Model {model_name} on dataset {dataset_name} was not found in the model wrapper registry'
            )
    return self._task_type_map[key]

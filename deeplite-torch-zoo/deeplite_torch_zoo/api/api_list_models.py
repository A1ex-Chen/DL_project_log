def list_models(filter_string='', print_table=True, task_type_filter=None,
    with_checkpoint=False):
    """
    A helper function to list all existing models or dataset calls
    It takes a `model_name` or a `dataset_name` as a filter and
    prints a table of corresponding available models

    :param filter: a string or list of strings containing model name, dataset name or "model_name_dataset_name"
    to use as a filter
    :param print_table: Whether to print a table with matched models (if False, return as a list)
    """
    if with_checkpoint:
        all_model_keys = MODEL_WRAPPER_REGISTRY.pretrained_models.keys()
    else:
        all_model_keys = MODEL_WRAPPER_REGISTRY.registry_dict.keys()
    if task_type_filter is not None:
        allowed_task_types = set(MODEL_WRAPPER_REGISTRY.task_type_map.values())
        if task_type_filter not in allowed_task_types:
            raise RuntimeError(
                f'Wrong task type filter value. Allowed values are {allowed_task_types}'
                )
        all_model_keys = [model_key for model_key in all_model_keys if 
            MODEL_WRAPPER_REGISTRY.task_type_map[model_key] == task_type_filter
            ]
    all_models = {(model_key.model_name + '_' + model_key.dataset_name):
        model_key for model_key in all_model_keys}
    models = []
    include_filters = filter_string if isinstance(filter_string, (tuple, list)
        ) else [filter_string]
    for f in include_filters:
        include_models = fnmatch.filter(all_models.keys(), f'*{f}*')
        if include_models:
            models = set(models).union(include_models)
    found_model_keys = [all_models[model] for model in sorted(models)]
    if not print_table:
        return found_model_keys
    table = texttable.Texttable()
    rows = collections.defaultdict(list)
    for model_key in found_model_keys:
        rows[model_key.model_name].extend([model_key.dataset_name])
    for model in rows:
        rows[model] = ', '.join(rows[model])
    table.add_rows([['Model name', 'Pretrained checkpoints on datasets'], *
        rows.items()])
    LOGGER.info(table.draw())
    return table

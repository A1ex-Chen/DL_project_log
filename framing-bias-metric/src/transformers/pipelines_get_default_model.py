def get_default_model(targeted_task: Dict, framework: Optional[str],
    task_options: Optional[Any]) ->str:
    """
    Select a default model to use for a given task. Defaults to pytorch if ambiguous.

    Args:
        targeted_task (:obj:`Dict` ):
           Dictionary representing the given task, that should contain default models

        framework (:obj:`str`, None)
           "pt", "tf" or None, representing a specific framework if it was specified, or None if we don't know yet.

        task_options (:obj:`Any`, None)
           Any further value required by the task to get fully specified, for instance (SRC, TGT) languages for
           translation task.

    Returns

        :obj:`str` The model string representing the default model for this pipeline
    """
    if is_torch_available() and not is_tf_available():
        framework = 'pt'
    elif is_tf_available() and not is_torch_available():
        framework = 'tf'
    defaults = targeted_task['default']
    if task_options:
        if task_options not in defaults:
            raise ValueError(
                'The task does not provide any default models for options {}'
                .format(task_options))
        default_models = defaults[task_options]['model']
    elif 'model' in defaults:
        default_models = targeted_task['default']['model']
    else:
        raise ValueError(
            'The task defaults can\'t be correctly selected. You probably meant "translation_XX_to_YY"'
            )
    if framework is None:
        framework = 'pt'
    return default_models[framework]

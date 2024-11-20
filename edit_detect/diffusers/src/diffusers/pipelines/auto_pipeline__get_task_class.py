def _get_task_class(mapping, pipeline_class_name, throw_error_if_not_exist:
    bool=True):

    def get_model(pipeline_class_name):
        for task_mapping in SUPPORTED_TASKS_MAPPINGS:
            for model_name, pipeline in task_mapping.items():
                if pipeline.__name__ == pipeline_class_name:
                    return model_name
    model_name = get_model(pipeline_class_name)
    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class
    if throw_error_if_not_exist:
        raise ValueError(
            f"AutoPipeline can't find a pipeline linked to {pipeline_class_name} for {model_name}"
            )

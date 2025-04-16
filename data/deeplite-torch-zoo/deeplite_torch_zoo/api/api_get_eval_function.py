def get_eval_function(model_name, dataset_name):
    task_type = MODEL_WRAPPER_REGISTRY.get_task_type(model_name=model_name,
        dataset_name=dataset_name)
    eval_function = EVAL_WRAPPER_REGISTRY.get(task_type=task_type,
        model_name=model_name, dataset_name=dataset_name)
    return eval_function

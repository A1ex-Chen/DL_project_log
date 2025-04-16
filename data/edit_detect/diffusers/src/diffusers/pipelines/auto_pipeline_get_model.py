def get_model(pipeline_class_name):
    for task_mapping in SUPPORTED_TASKS_MAPPINGS:
        for model_name, pipeline in task_mapping.items():
            if pipeline.__name__ == pipeline_class_name:
                return model_name

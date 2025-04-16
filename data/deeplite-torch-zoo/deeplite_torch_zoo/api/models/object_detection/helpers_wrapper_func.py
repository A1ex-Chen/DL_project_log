@MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=
    dataset_name, task_type='object_detection', has_checkpoint=has_checkpoint)
def wrapper_func(pretrained=False, num_classes=num_classes, **kwargs):
    default_kwargs.update(kwargs)
    return wrapper_generator_fn(model_name=model_name, dataset_name=
        dataset_name, num_classes=num_classes, pretrained=pretrained, **
        default_kwargs)
